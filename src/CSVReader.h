//
// Created by nishith on 1/23/18.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include "sstreamconvert.h"
#include "graph.h"

#ifndef TRIANGLE_COUNT_CSVREADER_H
#define TRIANGLE_COUNT_CSVREADER_H

std::vector<std::string> split(const char *phrase, std::string delimiter){
    std::vector<std::string> list;
    std::string s = std::string(phrase);
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    return list;
}

/*
 * A class to read data from a csv file.
 */
class CSVReader
{
public:
	std::string fileName;
	std::string delimeter;


	CSVReader(std::string filename, std::string delm = "\t") :
			fileName(filename), delimeter(delm)
	{ }

    // Function to fetch data from a CSV File
    /*
    * Parses through tsv file line by line and returns the data
    * in vector<edge<datatype> >.
    */
    template <typename dataType>
	std::vector<edge<dataType> >  getData()
    {
        std::ifstream file(fileName);
        std::vector<edge<dataType> > dataList;

        std::string line = "";
        // Iterate through each line and split the content using delimeter
        while (getline(file, line))
        {
            std::vector<std::string> vec;
            vec = split(line.data(), delimeter);
            // push back the first 2 data values
            edge<dataType> conv_vec(convert_to<dataType>(vec[0]), convert_to<dataType>(vec[1]));
            dataList.push_back(conv_vec);
        }
        // Close the File
        file.close();

        return dataList;
    }
};

#endif //TRIANGLE_COUNT_CSVREADER_H
