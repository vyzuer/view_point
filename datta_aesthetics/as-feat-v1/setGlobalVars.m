function setGlobalVars()
    global dataset_path dump_path image_list a_score_list image_src f_dump a_dump
    dataset_path = '/mnt/project/Datasets/DPChallenge/';
    dump_path = '/mnt/project/VP/datta_aesthetics/';
    image_list = strjoin({dataset_path, 'image.list'}, '/');
    a_score_list = strjoin({dataset_path, 'aesthetic.scores'}, '/');
    image_src = strjoin({dataset_path, 'ImageDB/'}, '/');
    f_dump = strjoin({dump_path, 'features.list'}, '/');
    a_dump = strjoin({dump_path, 'aesthetic.scores'}, '/');

    % dataset_path
    % dump_path
    % image_list
    % a_score_list
    % image_src
    % f_dump
    % a_dump

