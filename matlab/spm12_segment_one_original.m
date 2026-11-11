function spm12_segment_one_original(input_file, case_out, spm_root)
% Run SPM12 Segment for one original-grid T1 image.
% The input is only decompressed/copied; no external resampling is done.

if nargin ~= 3
    error('Usage: spm12_segment_one_original(input_file, case_out, spm_root)');
end

if exist(spm_root, 'dir') ~= 7 || exist(fullfile(spm_root, 'spm.m'), 'file') ~= 2
    error('Invalid SPM12 root: %s', spm_root);
end
if exist(input_file, 'file') ~= 2
    error('Input file not found: %s', input_file);
end
if exist(case_out, 'dir') ~= 7
    mkdir(case_out);
end

addpath(spm_root);
addpath(fullfile(spm_root, 'matlabbatch'));
addpath(fullfile(spm_root, 'config'));
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

[~, base, ext] = fileparts(input_file);
if strcmpi(ext, '.gz')
    [~, base_no_nii, ~] = fileparts(base);
    dst_nii = fullfile(case_out, [base_no_nii '.nii']);
    if exist(dst_nii, 'file') ~= 2
        gunzip(input_file, case_out);
        unpacked = fullfile(case_out, base);
        if exist(unpacked, 'file') ~= 2
            error('gunzip did not create expected file: %s', unpacked);
        end
        if ~strcmp(unpacked, dst_nii)
            movefile(unpacked, dst_nii, 'f');
        end
    end
else
    dst_nii = fullfile(case_out, [base ext]);
    if exist(dst_nii, 'file') ~= 2
        copyfile(input_file, dst_nii);
    end
end

[~, image_base, ~] = fileparts(dst_nii);
c1 = fullfile(case_out, ['c1' image_base '.nii']);
c2 = fullfile(case_out, ['c2' image_base '.nii']);
c3 = fullfile(case_out, ['c3' image_base '.nii']);
if exist(c1, 'file') == 2 && exist(c2, 'file') == 2 && exist(c3, 'file') == 2
    fprintf('[SPM12][SKIP] c1/c2/c3 already exist: %s\n', case_out);
    return;
end

tpm = fullfile(spm_root, 'tpm', 'TPM.nii');
if exist(tpm, 'file') ~= 2
    error('TPM.nii not found: %s', tpm);
end

matlabbatch = [];
matlabbatch{1}.spm.spatial.preproc.channel.vols = {[dst_nii ',1']};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 18;
% Match the 7T SPM batch: also write the bias-corrected native image.
matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];

ngaus = [1 1 2 3 4 2];
native = [1 0; 1 0; 1 0; 0 0; 0 0; 0 0];
for k = 1:6
    matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = {sprintf('%s,%d', tpm, k)};
    matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = ngaus(k);
    matlabbatch{1}.spm.spatial.preproc.tissue(k).native = native(k, :);
    matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = [0 0];
end

matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
% Match the 7T SPM batch internal sampling distance.
matlabbatch{1}.spm.spatial.preproc.warp.samp = 2;
matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];

fprintf('[SPM12][START] %s\n', dst_nii);
spm_jobman('run', matlabbatch);

if exist(c1, 'file') ~= 2 || exist(c2, 'file') ~= 2 || exist(c3, 'file') ~= 2
    error('SPM finished without all c1/c2/c3 outputs: %s', case_out);
end
fprintf('[SPM12][DONE] %s\n', case_out);
end
