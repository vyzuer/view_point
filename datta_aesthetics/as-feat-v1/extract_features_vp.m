function extract_features_vp()
    % set the global variables...
    dataset_path = '/home/vyzuer/View-Point/DataSet-VPF/';
    dump_path = '/mnt/project/VP/datta_aesthetics/VP-DUMPS/';

    % for loc = {'arcde', 'colognecathedral', 'merlion', 'taj', 'vatican', 'leaningtower', 'liberty', 'indiagate', 'gatewayofindia', 'eifel', 'forbiddencity', 'tiananmen'}
    % for loc = {'eifel', 'forbiddencity', 'tiananmen'}
    % for loc = {'taj', 'vatican', 'leaningtower'}
    for loc = {'liberty', 'indiagate', 'gatewayofindia'}

        location = char(loc);

        image_list = strjoin({dataset_path, char(location), 'image.list'}, '/');
        a_score_list = strjoin({dataset_path, location, 'aesthetic.scores'}, '/');
        image_src = strjoin({dataset_path, location, 'ImageDB/'}, '/');
        dump_dir = strjoin({dump_path, location}, '/');
        f_dump = strjoin({dump_path, location, 'features.list'}, '/');
        a_dump = strjoin({dump_path, location, 'aesthetic.scores'}, '/');

        % iterate thorugh the images and extract features

        if ~exist(dump_dir, 'dir')
            mkdir(dump_dir);
        end

        % if exist(f_dump, 'file')==2
        %     delete(f_dump);
        % end
        
        % imgs_list = load(image_list)

        str = fileread(image_list);
        imgs_list = textscan(str, '%s','delimiter', '\r');

        % fileID = fopen(a_score_list,'r');
        % a_scores = fscanf(fileID, '%f');

        copyfile(a_score_list, a_dump)

        for img = imgs_list{1}'
            % fprintf('%.6f\n', a_scores(i));
            infile = strjoin({image_src, char(img)}, '/');
            fprintf('%s\n', infile)

            getAestheticFeaturesVP(infile, f_dump);

        end

        % fclose(fileID);
    end

    return


