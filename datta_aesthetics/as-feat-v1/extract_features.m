function extract_features()
    % set the global variables...
    setGlobalVars()

    % iterate thorugh the images and extract features

    global dataset_path dump_path image_list image_src f_dump

    % if exist(f_dump, 'file')==2
    %     delete(f_dump);
    % end
    
    % imgs_list = load(image_list)

    str = fileread(image_list);
    imgs_list = textscan(str, '%s','delimiter', '\r');

    % fileID = fopen(a_score_list,'r');
    % a_scores = fscanf(fileID, '%f');

    for img = imgs_list{1}'
        % fprintf('%.6f\n', a_scores(i));
        infile = strjoin({image_src, char(img)}, '/');
        fprintf('%s\n', infile)

        getAestheticFeaturesVP(infile, f_dump);

    end

    % fclose(fileID);

