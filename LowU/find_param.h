#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<boost/algorithm/string.hpp>

int find_param(std::string file_name, std::string param, double &value)
{
	std::string line, data;
	std::vector <std::string> pv;
	std::ifstream file(file_name.c_str());
	while(std::getline(file, line))
	{
		if (line.find_first_not_of("\t\n ") != std::string::npos)
		{
			boost::split(pv, line, [](char s){return s == '=';});
			if (pv[0] == param)
			{
				value = stod(pv[1]);
				return 0;
			}
		}
	}
	return 1;
}

int find_param(std::string file_name, std::string param, int &value)
{
	std::string line, data;
	std::vector <std::string> pv;
	std::ifstream file(file_name.c_str());
	while(std::getline(file, line))
	{
		if (line.find_first_not_of("\t\n ") != std::string::npos)
		{
			boost::split(pv, line, [](char s){return s == '=';});
			if (pv[0] == param)
			{
				value = stoi(pv[1]);
				return 0;
			}
		}
	}
	return 1;
}
