# Real-Time 2D object Recognition
-------------------------------------------------------
## File structure
keep the following files in same folder
- CMakeList.txt
- main.cpp
- features.cpp
- csv_util.cpp
- features.h
- csv_util.h
-----------------------------------------------------------------------------------------------
In order to run the code execute following commands
```sh
~ cmake .
~ make
```
which will create the executable file in a bin folder which can be executed by following command

```sh
~ cd /bin
~ ./main <csv file name> <classifier>
```
----------------------------------------------------------------------------------------------
In the system there are 2 modes

1.Inference Mode 
2.training Mode

When 'T' is pressed it enters the training Mode and performs the prepossing for current frame and lets user enter the (new/old) class name and save it in database

when pressed again it goes back to Inference Mode

when 'q' is pressed it quits the system
