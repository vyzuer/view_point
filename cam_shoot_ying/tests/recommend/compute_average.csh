#!/bin/csh 

set dump_path = /mnt/project/VP/cam_shoot_ying/quality_dumps/VP-DUMPS/
set num_locations = 12

pushd $dump_path > /dev/null

set b_p1 = `cat */basic.ndcg | sed 's/^n1.*$//' | cut -f 1 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p1 : $b_p1"

set b_p2 = `cat */basic.ndcg | sed 's/^n1.*$//' | cut -f 2 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p2 : $b_p2"

set b_p5 = `cat */basic.ndcg | sed 's/^n1.*$//' | cut -f 3 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p5 : $b_p5"

set b_n1 = `cat */basic.ndcg | sed 's/^p1.*$//' | cut -f 1 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n1 : $b_n1"

set b_n2 = `cat */basic.ndcg | sed 's/^p1.*$//' | cut -f 2 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n2 : $b_n2"

set b_n5 = `cat */basic.ndcg | sed 's/^p1.*$//' | cut -f 3 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n5 : $b_n5"

set t_p1 = `cat */time.ndcg | sed 's/^n1.*$//' | cut -f 1 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p1 : $t_p1"

set t_p2 = `cat */time.ndcg | sed 's/^n1.*$//' | cut -f 2 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p2 : $t_p2"

set t_p5 = `cat */time.ndcg | sed 's/^n1.*$//' | cut -f 3 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "p5 : $t_p5"

set t_n1 = `cat */time.ndcg | sed 's/^p1.*$//' | cut -f 1 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n1 : $t_n1"

set t_n2 = `cat */time.ndcg | sed 's/^p1.*$//' | cut -f 2 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n2 : $t_n2"

set t_n5 = `cat */time.ndcg | sed 's/^p1.*$//' | cut -f 3 | cut -d' ' -f 2 | awk '{ sum+=$1} END {print sum/'"${num_locations}"'}'`
echo "n5 : $t_n5"

cd ../

echo "Basic" > "average.scores"
echo "p1:$b_p1 p2:$b_p2 p5:$b_p5" >> "average.scores"
echo "n1:$b_n1 n2:$b_n2 n5:$b_n5" >> "average.scores"

echo "Time" >> "average.scores"
echo "p1:$t_p1 p2:$t_p2 p5:$t_p5" >> "average.scores"
echo "n1:$t_n1 n2:$t_n2 n5:$t_n5" >> "average.scores"

cd -

popd > /dev/null

