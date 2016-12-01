#!/bin/csh

foreach line ( "`cat /etc/passwd`" )
    echo $line
   set line = "$line:gas/ /_/"
   set line = "$line:gas/:/ /"
   set argv = ( $line )
   set name1 = $1
   set name2 = "$5:gas/_/ /"
   echo $name2
end
