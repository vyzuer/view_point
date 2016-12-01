#!/bin/csh -f
#wget "http://www.wunderground.com/history/airport/WSSS/2013/3/20/DailyHistory.html?req_city=Singapore&req_state=&req_statename=Singapore&format=1"
#wget "http://www.weatherbase.com/weather/weatherhourly.php3?s=89684&date=2013-03-05&cityname=Singapore%2C+Singapore%2C+Singapore"
#wget "http://www.weatherbase.com/weather/weatherhourly.php3?s=89684&date=2013-03-01&cityname=Singapore%2C+Singapore%2C+Singapore&set=us"

set url_1 = "http://www.wunderground.com/history/airport/WSSS/"
set url_2 = "/DailyHistory.html"
set url_3 = "req_city=Singapore&req_state=&req_statename=Singapore&format=1"

set years = 2000
set months = 12
set days = 31

set year = 2013
mkdir -p WDB2
pushd WDB2

while ($year >= $years)
    mkdir -p $year
    pushd $year
    set month = 1
    while($month <= $months)
        mkdir -p $month
        pushd $month
        set day = 1
        while($day <= $days)

            set d = `echo $day | awk '{printf "%02d\n", $0;}'`
            set m = `echo $month | awk '{printf "%02d\n", $0;}'`
            set urlDate = "${year}/${m}/${d}"
            set fName = "${day}.info"
#            echo $url
            echo $fName
            wget -t 2 "${url_1}${urlDate}${url_2}?${url_3}" -O $fName
            @ day++

        end
        sleep 35
        @ month++
        popd
    end
    @ year--
    popd
end

popd
