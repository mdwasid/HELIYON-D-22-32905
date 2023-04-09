%% Adaptive Genetic Algorithm for User Preference Discovery in Multi-Criteria Recommender Systems

%% basic screen clearing commands
% clc; close all; clear;

% GA parameter initialization
pop_size=10;  runs=10; generation=30;

neigh_size=input('Enter the neighbour size eg., 10, 20, 30, ...');  %select size of neighorhood i.e. top-K
%% importing the datset
filename='userxitem_YM.xlsx';
userxitem_db=xlsread(filename); % 484*945 for Yahoo! Movies

db_size = size(userxitem_db);
%% Mean of each user
User_mean =  sum(userxitem_db,2)./sum(userxitem_db ~=0,2);

ye=0;
for train_test=.60:.10:.90   %training split ratio 
    ye=ye+1;
    train_size= zeros(1,db_size(1));
    total_MAE=zeros(1,db_size(1)); total_RMSE= zeros(1,db_size(1)); total_coverage= zeros(1,db_size(1)); total_accuracy= zeros(1,db_size(1));
    total_precision= zeros(1,db_size(1)); total_recall = zeros(1,db_size(1)); total_fmeasure = zeros(1,db_size(1));
    
    actuser_mae=zeros(db_size(1),runs);
    actuser_rmse=zeros(db_size(1),runs);
    accuracy=zeros(db_size(1),runs);
    precision=zeros(db_size(1),runs);
    recall=zeros(db_size(1),runs);
    fmeasure=zeros(db_size(1),runs);
    %%
    for actuser=1:db_size(1) %484 number of users
        %     sim1 = zeros(1,db_size(1)); % Preallocating the memory
        %     sim = zeros(db_size(1),1); % Preallocating the memory for the similarity
        
        %% Random initial population
        chromo= randi([0,1],pop_size,32);     % 8X4=32 bit chromosome binary encoding
        
        %% Generation of training and test data
        new_db = find(userxitem_db(actuser,:)); %for user one ## at a time one user's rating
        train_size(1,actuser) = round(length(new_db)*train_test);  %70% as training set
        ni = length(new_db)- train_size(1,actuser);
        train_index = new_db(:,1:train_size(1,actuser));
        test_index  = new_db(:,train_size(1,actuser)+1 : end);
        train_rat =zeros(1,train_size(1,actuser));
        for a=1:train_size(1,actuser)
            train_rat(1,a) = userxitem_db(actuser,train_index(a));
        end
        actuser_mean =  mean(train_rat); % udate the mean of active user only
        %% call function - Learning Weights Using GA
        [new_fit,neighbour,dim] = actuserfit(actuser,freq_count,chromo,userxitem_db,train_index,User_mean,pop_size,neigh_size,train_size,db_size(1),train_test);
        sort_fit=sortrows(new_fit,1); %Increasing order fitness
        top_chrom(:,1)= sort_fit(:,2);
        old_fit = sort_fit(:,1);
        for y=1:pop_size
            old_fit(y,2)=y+10;
            chromo1(y,:) = chromo(top_chrom(y),:);
        end
        
        %% System run
        fit=1;
        for use=1:runs
            iteration=1;
            while (iteration <=generation) && (fit >= 0.025)
                [new_chromo]= cross_mut(chromo1); %function call -- apply crossover - mutation operators
                [new_fit,neighbour,dim] = actuserfit(actuser,freq_count,new_chromo,userxitem_db,train_index,User_mean,pop_size,neigh_size,train_size,db_size(1),train_test);
                chromo = [new_chromo; chromo1 ];
                full_fit  = [new_fit; old_fit ];
                fit=min(full_fit(:,1));
                sort_fit=sortrows(full_fit,1); %Dicreasing order fitness
                top_chrom(1:10,1)= sort_fit(1:10,2);
                old_fit=sort_fit(1:10,1);
                for y=1:pop_size
                    old_fit(y,2)=y+10;
                    chromo1(y,:)=chromo(top_chrom(y),:);
                end
                iteration= iteration+1;
            end
            [minfit,ind] = min(new_fit(:,1));  % ######  Best Fitness
            
            %% Prediction
            pi = 0; prediction=zeros(1,ni);
            for pr=1:ni            % movies to predict by neighbourhood set
                right = 0; norm_k = 0; count=0; % count how many neighbours are able to give prediction
                for n=1:neigh_size             % nighbours
                    neigh =neighbour(n,ind);
                    if  userxitem_db(neigh,test_index(1,pr))~=0
                        right = right + (dim(neigh,ind) * (userxitem_db(neigh,test_index(1,pr)) - User_mean(neigh)));
                        norm_k = norm_k + abs(dim(neigh,ind));
                        count = count+1;
                    end
                end
                if count ~=0 || (right ~=0 && norm_k ~= 0)
                    K = 1/norm_k;   %normalization factor
                    prediction(pr) = User_mean(actuser)+ (K * right);  % round use to calculate correct pred
                    pi=pi+1;
                else
                    prediction(pr) = 0;
                end
            end
            
            %% Coverage, MAE and RMSE of actuser
            
            actuser_pi(actuser,use)=pi;  % total number of predicted items for actuser
            actuser_ni(actuser)=ni;
            
            mae=0; rmse=0;
            for item=1:ni
                if prediction(item) ~= 0
                    mae= mae + abs(prediction(item) - userxitem_db(actuser,test_index(1,item)));
                    rmse= rmse + (abs(prediction(item) - userxitem_db(actuser,test_index(1,item))))^2;
                end
            end
            if pi ~=0
                actuser_mae(actuser,use)=mae/pi;
                actuser_rmse(actuser,use)=sqrt(rmse/pi);
            else
                actuser_mae(actuser,use)=0;
                actuser_rmse(actuser,use)=0;
            end
            
            
            %% Precision,Recall & F-measure
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
                accuracy(actuser,use) = (tp+tn)/ (tp+tn+fp+fn);
            end
            if (tp+fp) ~=0
                precision(actuser,use) = tp/(tp+fp);
            end
            if (tp+fn)~=0
                recall(actuser,use)=tp/(tp+fn);
            end
            if recall(actuser,use)~=0 || precision(actuser,use) ~=0
                fmeasure(actuser,use)=(2*precision(actuser,use)*recall(actuser,use))/(precision(actuser,use)+recall(actuser,use));
            end
        end
        %     total1_coverage(y,:)= (round((sum(actuser_pi)./sum(actuser_ni))*10000))/10000;
        total_MAE(actuser)= min(actuser_mae(actuser,:));
        total_RMSE(actuser)= min(actuser_rmse(actuser,:));
        total_coverage(actuser) = max(actuser_pi(actuser,:));
        total_accuracy(actuser) = max(accuracy(actuser,:));
        total_precision(actuser) = max(precision(actuser,:));
        total_recall(actuser) = max(recall(actuser,:));
        total_fmeasure(actuser) = max(fmeasure(actuser,:));
        
    end
    total1_coverage(ye,:)= (round((sum(total_coverage)./sum(actuser_ni))*10000))/10000;
    total2_MAE(ye,:)=(round((sum(total_MAE)./sum(total_MAE ~=0))*10000))/10000;
    total3_RMSE(ye,:)=(round((sum(total_RMSE)./sum(total_RMSE ~=0))*10000))/10000;
    total4_precision(ye,:) = (round((sum(total_precision)./sum(total_precision ~=0))*10000))/10000;
    total5_recall(ye,:)= (round((sum(total_recall)./sum(total_recall ~=0))*10000))/10000;
    total6_fm(ye,:)= (round((sum(total_fmeasure)./sum(total_fmeasure ~=0))*10000))/10000;
    total7_accuracy(ye,:)=(round((sum(total_accuracy)./sum(total_accuracy ~=0))*10000))/10000;
end
gap=[0];
Grand_result = [total1_coverage; gap; total2_MAE;gap;total3_RMSE; gap;total4_precision;gap; total5_recall;gap;total6_fm;gap;total7_accuracy];
