%% Clustering with Side Information approach (Clust-SI)

%% basic screen clearing commands
clc; close all; clear;

%% importing the datset
filename='userxitem_YM.xlsx';
userxitem_db=xlsread(filename);     % 484*945 for Yahoo! Movies
filename='Clusters_SI.xlsx'; 
% Clusters obtained through clustering users through side information

cluster=xlsread(filename);
db_size = size(userxitem_db);
%% Mean of each user
User_mean =  sum(userxitem_db,2)./sum(userxitem_db ~=0,2);
%% Generation of training and test data
y=0;
for train_test=.60:.10:.90
    y=y+1; precision=zeros(db_size(1),1); accuracy=zeros(db_size(1),1);
    recall=zeros(db_size(1),1); fmeasure=zeros(db_size(1),1); actuser_pi=zeros(db_size(1),1); actuser_ni=zeros(db_size(1),1);
    actuser_mae=zeros(db_size(1),1); actuser_rmse=zeros(db_size(1),1); actuser_corpred=zeros(db_size(1),1);
    
    for actuser=1 :db_size(1)
        [r1,c1]=find(cluster==actuser);
        len=length(find(cluster(:,c1)~=0));
        c_users=cluster(1:len,c1);
        sim=[];
        %sim = zeros(db_size(1),1); % Preallocating the memory for the similarity
        for g = 1:len
            if c_users(g) ~= actuser
                new_db = find(userxitem_db(actuser,:)); %for user one ## at a time one user's rating
                train_size(1,actuser) = round(length(new_db)*train_test);  % training set
                ni = length(new_db)- train_size(1,actuser);
                train_index = new_db(:,1:train_size(1,actuser));
                test_index  = new_db(:,train_size(1,actuser)+1 : end);
                train =zeros(1,train_size(1,actuser));
                for a=1:train_size(1,actuser)
                    train(1,a) = userxitem_db(actuser,train_index(a));
                end
                actuser_mean =  mean(train); % udate the mean of active user 
                %% Caclulating the similarity among different users
                comm=0; maha_dis=[];  %number of co-rated items given by user u and v
                for d = 1: train_size(1,actuser)
                    if  userxitem_db(c_users(g),train_index(d)) ~=0
                        mat(:,1)=userxitem_db(actuser,:);       
                        mat(:,2)=userxitem_db(c_users(g),:);
                        cov_mat=cov(mat);
                        sigma1=sqrt(cov_mat(1,1));
                        sigma2=sqrt(cov_mat(2,2));
                        raw12=(cov_mat(1,2)/(sigma1*sigma2));
                        temp=(train(d)- actuser_mean)/sigma1;
                        maha_dis(d)=sqrt(temp^2 + ((((userxitem_db(c_users(g),train_index(d))- User_mean(c_users(g)))/sigma2)-(raw12*temp))*(1/(sqrt(1-raw12^2))))^2);
                        comm=comm+1;
                    end
                end
                if comm ~=0
                    sim(g,1)=sum(maha_dis)/comm; %distance between actuser and user g
                else
                    sim(g,1)=999;
                end
            else
                sim(g,1)= 999;                  %highest possible distance for self distance
            end
        end
        %% sort the sim array
        sim(:,2)=c_users;
        sim((find(sim(:,1)==0)),1)=999;
        sortsim=sortrows(sim,1);
        x=0;
        for top_k=10:10:70
            x=x+1;
            neighbour = sortsim(1:top_k,2);    %topK neighbours
            %% Prediction
            prediction=zeros(1,ni);
            pi = 0;
            for pr=1:ni            % movies to predict by neighbourhood set
                right = 0; norm_k = 0; count=0; % count the number of neighbours are able to give prediction            
                for n=1:top_k             % nighbours
                    f=find(sim(:,2)==neighbour(n,1));
                    if  userxitem_db(neighbour(n,1),test_index(1,pr))~=0 % && userxitem_db(actuser,test_index(1,pr)) ~=0  #### already non-zero coz it's in test_index
                        right = right + (sim(f,1) * (userxitem_db(neighbour(n,1),test_index(1,pr)) - User_mean(neighbour(n,1))));
                        norm_k = norm_k + abs(sim(f,1));
                        count = count+1;
                    end
                end
                
                if count ~=0 || (right ~=0 && norm_k ~= 0)
                    
                    prediction(pr) =actuser_mean+ ( right/norm_k);
                    pi=pi+1;
                else
                    prediction(pr) = 0;
                end
            end
            actuser_pi(actuser,x)=pi;  % total number of predicted items for actuser
            actuser_ni(actuser,x)=ni;   % total number of items in test set
            %% Coverage, MAE and RMSE of actuser
            mae=0; correct =0; rmse=0;
            for  item=1:ni
                if prediction(item) ~= 0
                    if round(prediction(item)) == userxitem_db(actuser,test_index(1,item))
                        correct = correct+1;
                    end
                    mae= mae + abs( prediction(item) - userxitem_db(actuser,test_index(1,item)));
                    rmse= rmse + ( prediction(item) - userxitem_db(actuser,test_index(1,item)))^2;
                end
            end
            if pi ~=0
                actuser_mae(actuser,x)=mae/pi;
                actuser_rmse(actuser,x)=sqrt(rmse/pi);
                actuser_corpred(actuser,x)=correct/pi;
            else
                actuser_mae(actuser,x)=0;
                actuser_rmse(actuser,x)=0;
                actuser_corpred(actuser,x)=0;
            end
            %% Precision,Recall & F-measure Accuracy 
            thres=3; pr_size = size(prediction);
            tp=0;fn=0;fp=0;tn=0;
            for l=1:pr_size(2)
                p= prediction(l);
                a=userxitem_db(actuser,test_index(1,l));
                if (p~=0) && (a~=0)
                    if (p >= thres) && (a >= thres)
                        tp=tp+1;
                    elseif (a>p && a>=thres)
                        fn=fn+1;
                    elseif (p>a && p>=thres)
                        fp=fp+1;
                    else
                        tn=tn+1;
                    end
                end
            end
            
            if (tp+tn+fp+fn)~=0
                accuracy(actuser,x) = (tp+tn)/ (tp+tn+fp+fn);
            end
            if (tp+fp) ~=0
                precision(actuser,x) = tp/(tp+fp);
            end
            if (tp+fn)~=0
                recall(actuser,x)=tp/(tp+fn);
            end
            if recall(actuser,x)~=0 || precision(actuser,x) ~=0
                fmeasure(actuser,x)=(2*precision(actuser,x)*recall(actuser,x))/(precision(actuser,x)+recall(actuser,x));
            end
        end
    end
    total1_coverage(y,:)= (round((sum(actuser_pi)./sum(actuser_ni))*10000))/10000;
    total2_MAE(y,:)= (round((sum(actuser_mae)./sum(actuser_mae ~=0))*10000))/10000;
    total3_RMSE(y,:)=(round((sum(actuser_rmse)./sum(actuser_rmse ~=0))*10000))/10000;
    total4_precision(y,:) = (round((sum(precision)./sum(precision ~=0))*10000))/10000;
    total5_recall(y,:)= (round((sum(recall)./sum(recall ~=0))*10000))/10000;
    total6_fm(y,:)= (round((sum(fmeasure)./sum(fmeasure ~=0))*10000))/10000;
    total7_accuracy(y,:)=(round((sum(accuracy)./sum(accuracy ~=0))*10000))/10000;
    total8_correct(y,:)=(round((sum(actuser_corpred)./sum(actuser_corpred ~=0))*10000))/10000;
    
end
gap=[0,0,0,0,0,0,0];
Grand_result = [total1_coverage; gap; total2_MAE;gap;total3_RMSE; gap;total4_precision;gap; total5_recall;gap;total6_fm;gap;total7_accuracy;gap;total8_correct];
