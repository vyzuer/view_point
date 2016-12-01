#!/bin/csh -f

# set db_path = /mnt/windows/DataSet-YsR/
set db_path = /mnt/windows/DataSet-VPF/

pushd $db_path
# foreach location (${db_path}/*)
foreach location (forbiddencity tiananmen)

    pushd $location
    echo $location
    set years = 2000
    set months = 12
    
    set year = 2015
    pushd SunMoonDB
    
    while ($year >= $years)
        pushd $year
        set month = 1
        while($month <= $months)
            pushd $month
            set fName1 = "sun.info"
            echo $fName1
    
            set fName2 = "moon.info"
            echo $fName2
            html2text -ascii $fName1 > "sun_1.info"
            html2text -ascii $fName2 > "moon_1.info"
            popd
            @ month++
        end
        @ year--
        popd
    end
    
    popd
    popd

end

popd
