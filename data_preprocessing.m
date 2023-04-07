%% excution to be done step by step after inporting the dataset

%% importing the datset
filename='data_movies.xlsx';    %improt Yahoo! movies dataset
data_movies=xlsread(filename);

%% Generating user X item matrix
user_item=zeros(6078,976);
for sw=1:62156
    stop =4*data_movies(sw,7);
    start= stop-3;
    user_item(data_movies(sw,1),data_movies(sw,7)) = data_movies(sw,6);
    item_cri(data_movies(sw,1),start:stop) = data_movies(sw,2:5);
end

%% preprocessing user x item matrix
users= unique(user_item(:,1));
u=1; t=1; q=1;
for i=1 : length(users)
    count= find(user_item(:,1) == users(i));
    if length(count) > x     %identify users who have rated atleast x movies
        real_users(1,t)=users(i);
        real_users(2,t)=length(count);
        t=t+1;
    else
        fake_users(1,q)=users(i);  %identify users who do not satify the condition
        fake_users(2,q)=length(count);
        q=q+1;
    end
end


for j=1:length(fake_users)
    loc=find(new_db(:,1)== fake_users(1,j));
    new_db(loc,:)=[];           %remove users who do not satify the condition 
end


for j=1: length(count)
    x = user_item(count(j),2:3);
    userxitem_db(i,x(1)) =  x(2);
    %      end
    
end

% filename='User_info.xlsx';
% User_info=xlsread(filename);
% %% occurances
% check= zeros(100000 , 3);
% for i=1 : 100000
%     x = data_movies(i, 1);
%     check(i,:) = sum(data_movies == x);
% end
% t=1; s=1; satisfy= zeros(84596,1); unsatisfy= zeros(84596,1);
% for i=1 : 100000
%     if check(i,1) >=60
%     satisfy(t,1) = data_movies(i,1);
%     t=t+1;
%
%     else
%         unsatisfy(s,1)= data_movies(i,1);
%         s=s+1;
%     end
% end
%     total_user = unique(satisfy);
%     total_unsatisfy = unique(unsatisfy);
%    total_unsatisfy(1,1) = [];
% final_set = zeros(84596,3); t=1;
% for i=1 : 100000
%     for j = 1 : 497
%         if data_movies(i,1) == total_user(j,1)
%         final_set(t,:) = data_movies(i, :);
%         t= t+1;
%         end
%     end
% end
