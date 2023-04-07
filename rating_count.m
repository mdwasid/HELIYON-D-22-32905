%% basic screen clearing commands
clc; close all; clear;
%% importing the datset
filename='userxitem_YM.xlsx';
userxitem_db=xlsread(filename);

db_size = size(userxitem_db); 
%% Numner of items and criteria's rated
for i=1:db_size(1)
    pe(i,1)=length(find(userxitem_db(i,:)));		%number of non-zero ratings in overall ratings
    pe(i,2)=length(find(item_cri(i,:)))/4;			%number of non-zero ratings in multi-criteria ratings
end
for j=1:db_size(1)
    if pe(j,1)~=pe(j,2)
        pe(j,3)=999; % put 999 if #items is not equal to #criteria's
    end
end
%% generation of rating count matrix
freq=zeros(db_size(1),20);
for m=1:db_size(1)
    for k=1:4
        for t=k:4:size(item_cri,2)
            if item_cri(m,t) ~=0
                temp=(k-1)*5;
                freq(m,temp+item_cri(m,t))=freq(m,temp+item_cri(m,t))+1;
            end
        end
    end
end
%% dividing occurances with #movies
for k=1:db_size(1)
    pe(k,3)=sum(freq(k,:))/4;
    freq_count(k,:)=freq(k,:)/pe(k,3);			%rating count profile of each user of size 1X20
end



