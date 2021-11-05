%% Initial Commands
% The following code reads in the abalone dataset.
cdir = fileparts(mfilename('fullpath'));
% Since comma is used in the data, we need to convert these to dots
Data = fileread(fullfile(cdir,'../project/Data_abalone.csv'));
Data = strrep(Data, ',', '.');
FID = fopen('Data_abalone_dot.csv', 'w');
fwrite(FID, Data, 'char');
fclose(FID);
file_path = fullfile(cdir,'../project/Data_abalone_dot.csv');
abalone_table = readtable(file_path);
X = table2array(abalone_table(1:end, 2:9)); % Extracted only some data
%X = rmoutliers(X); %remove outliers123
% Table is now stored in Matlab
[NUMERIC, TXT, RAW] = xlsread(fullfile(cdir,'../project/Data_abalone.csv'));
abalone_table = readtable(file_path);
% Extract attribute names from the first column
attributeNames = RAW(1,1:end);
% Extract unique class names from the first row
classLabels = RAW(2:end,1);
classNames = unique(classLabels);
% Extract class labels that match the class names
[y_,y] = ismember(classLabels, classNames); y = y-1;

% Stadardize de data by substracting the mean and diving by the standard
% deviation. 
%X = bsxfun(@minus, X, mean(X));
%X = bsxfun(@times, X, 1./std(X));
% Removing the outliers. 
X = rmoutliers(X);
first_time=1;