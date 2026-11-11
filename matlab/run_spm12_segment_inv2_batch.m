%% Batch SPM12 unified segmentation for INV2 images.
% Outputs native GM/WM/CSF probability maps (c1/c2/c3) and bias-corrected INV2.
%
% Edit spmRoot, inRoot, and outRoot below before running this optional
% standalone MATLAB batch. The recommended parameterized entry point is
% powershell/run_spm12_pipeline.ps1.

clear; clc;

spmRoot = 'D:\BaiduNetdiskDownload\fsl\code\code\spm12-main\spm12-main';
inRoot = 'D:\BaiduNetdiskDownload\fsl\data\N4';
outRoot = 'D:\BaiduNetdiskDownload\fsl\data\spm12_segment_wm';

if exist(fullfile(spmRoot, 'spm.m'), 'file') ~= 2
    error('Cannot find spm.m under: %s', spmRoot);
end
if exist(inRoot, 'dir') ~= 7
    error('Cannot find input root: %s', inRoot);
end
if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

addpath(spmRoot);
addpath(fullfile(spmRoot, 'matlabbatch'));
addpath(fullfile(spmRoot, 'config'));
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

tpm = fullfile(spmRoot, 'tpm', 'TPM.nii');
if exist(tpm, 'file') ~= 2
    error('Cannot find TPM.nii: %s', tpm);
end

files = [dir(fullfile(inRoot, '*INV2*.nii')); dir(fullfile(inRoot, '*INV2*.nii.gz'))];
if isempty(files)
    error('No *INV2*.nii or *INV2*.nii.gz files found under: %s', inRoot);
end

fprintf('[SPM12] Found %d INV2 files under %s\n', numel(files), inRoot);

for i = 1:numel(files)
    src = fullfile(files(i).folder, files(i).name);
    [~, name, ext] = fileparts(src);

    % For .nii.gz, fileparts returns name='sub-xxx_INV2.nii', ext='.gz'.
    if strcmpi(ext, '.gz')
        niiName = name;
        [~, baseNoNii, ~] = fileparts(niiName);
    else
        niiName = [name ext];
        baseNoNii = name;
    end

    token = regexp(baseNoNii, '(sub-\d+)', 'tokens', 'once');
    if isempty(token)
        cid = baseNoNii;
    else
        cid = token{1};
    end

    fprintf('\n===== SPM12 segment %s (%d/%d) =====\n', cid, i, numel(files));
    caseOut = fullfile(outRoot, cid);
    if exist(caseOut, 'dir') ~= 7
        mkdir(caseOut);
    end

    dstNii = fullfile(caseOut, [baseNoNii '.nii']);
    if strcmpi(ext, '.gz')
        if exist(dstNii, 'file') ~= 2
            fprintf('[SPM12] gunzip %s -> %s\n', src, caseOut);
            gunzip(src, caseOut);
            unzipped = fullfile(caseOut, niiName);
            if ~strcmp(unzipped, dstNii)
                movefile(unzipped, dstNii);
            end
        else
            fprintf('[SPM12] existing decompressed file: %s\n', dstNii);
        end
    else
        if exist(dstNii, 'file') ~= 2
            copyfile(src, dstNii);
        end
    end

    if exist(dstNii, 'file') ~= 2
        warning('[SPM12] missing prepared NIfTI for %s, skip', cid);
        continue;
    end

    matlabbatch = [];
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[dstNii ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 18;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];

    tissue_ngaus = [1 1 2 3 4 2];
    tissue_native = [1 0; 1 0; 1 0; 0 0; 0 0; 0 0];
    for k = 1:6
        matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm = {sprintf('%s,%d', tpm, k)};
        matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus = tissue_ngaus(k);
        matlabbatch{1}.spm.spatial.preproc.tissue(k).native = tissue_native(k, :);
        matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = [0 0];
    end

    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 2;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];

    spm_jobman('run', matlabbatch);
end

fprintf('\n[SPM12] Done. Outputs are under: %s\n', outRoot);
