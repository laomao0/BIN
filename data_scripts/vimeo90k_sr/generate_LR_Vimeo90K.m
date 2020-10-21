function generate_LR_Vimeo90K()
    %% matlab code to genetate bicubic-downsampled for Vimeo90K dataset
    
    up_scale = 4;
    mod_scale = 4;
    idx = 0;
    imgDataPath = '/Vimeo90k_SR/vimeo_septuplet/sequences/';
    imgDataDir = dir(imgDataPath);     % travel all subdirs
    imgDataSubDir = imgDataDir(3:end);  % clear implimt firs 
    numSubdir = length(imgDataSubDir);
    
    
    save_LR_folder = strcat('/Vimeo90k_SR/','vimeo_septuplet_matlabLRx4');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    
    save_LR_folder = strcat('/Vimeo90k_SR/','vimeo_septuplet_matlabLRx4', '/', 'sequences');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    
    
    for vnumSubDir=1:numSubdir
        subDirPath = strcat(imgDataPath,imgDataSubDir(vnumSubDir).name,'/');
        imgDataSubSubDir = dir(subDirPath);
        imgDataSubSubDir = imgDataSubSubDir(3:end);
        numSubSubDir = length(imgDataSubSubDir);
    
        save_LR_sub_folder = strcat(save_LR_folder,'/',imgDataSubDir(vnumSubDir).name);
        if ~exist(save_LR_sub_folder, 'dir')
            mkdir(save_LR_sub_folder);
        end
    
        for vnumSubSubDir=1:numSubSubDir
    
            subsubDirPath = strcat(subDirPath, imgDataSubSubDir(vnumSubSubDir).name, '/', '*.png');
            filepaths = dir(subsubDirPath);
    
            save_LR_subsub_folder = strcat(save_LR_sub_folder,'/',imgDataSubSubDir(vnumSubSubDir).name);
            if ~exist(save_LR_subsub_folder, 'dir')
                mkdir(save_LR_subsub_folder);
            end
    
            for i = 1 : length(filepaths)
                [~,imname,ext] = fileparts(filepaths(i).name);
                folder_path = strcat(subDirPath, imgDataSubSubDir(vnumSubSubDir).name);
    
    
                if isempty(imname)
                    disp('Ignore . folder.');
                elseif strcmp(imname, '.')
                    disp('Ignore .. folder.');
                else
                    idx = idx + 1;
                    str_rlt = sprintf('%d\t%s.\n', idx, imname);
                    fprintf(str_rlt);
                    % read image
                    img = imread(fullfile(folder_path, [imname, ext]));
                    img = im2double(img);
                    % modcrop
                    img = modcrop(img, mod_scale);
                    % LR
                    im_LR = imresize(img, 1/up_scale, 'bicubic');
                    if exist('save_LR_subsub_folder', 'var')
                        imwrite(im_LR, fullfile(save_LR_subsub_folder, [imname, '.png']));
                    end
                end
            end
        end
    
    
    
    
    end
    end
    
    %% modcrop
    function img = modcrop(img, modulo)
    if size(img,3) == 1
        sz = size(img);
        sz = sz - mod(sz, modulo);
        img = img(1:sz(1), 1:sz(2));
    else
        tmpsz = size(img);
        sz = tmpsz(1:2);
        sz = sz - mod(sz, modulo);
        img = img(1:sz(1), 1:sz(2),:);
    end
    end
    