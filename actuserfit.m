function[actuser_fit1,neighbour,over_sim] = actuserfit(actuser,freq_count,chromo,userxitem_db,train_index1,User_mean,v,neigh_size,train_size,db_size,train_test)

%% Initial weight of each criteria
weight=zeros(v,4);

for tx=1:v
    q=0;
    for ty=1:4
        p=q+1; q=q+8;
        position(tx,ty)=bin2dec(num2str(chromo(tx,p:q)));
    end
end
for s=1:v
    add = sum(position(s,:));
    for t=1:4
        weight(s,t)=position(s,t)/add;  %initial criteria weights
    end
end
%% Local Similarity from Multi-Criteria Ratings
sim = zeros(db_size,1); crsim= zeros(db_size,1);% Preallocating the memory for the similarity
for ch=1:v  %for each chromosone
    for g = 1:db_size
        if g ~= actuser
            sim2=0; pos=1;
            for p=1:5:20
                sim1=0;
                for k=p:p+4
                    sim1 =sim1+ weight(ch,pos)*(freq_count(actuser,k)-freq_count(g,k))^2;
                end
                sim2=sim2+sqrt(sim1); pos=pos+1;
            end
            sim(g,ch)=1/(1+(sim2/4));
        else
            sim(g,ch)=0;
        end
    end
end

%% Local Similarity from Overall Ratings
for g = 1:db_size
    if g ~= actuser
        train_n= find(userxitem_db(g,:));
        comm_rat=length(intersect(train_index1,train_n));
        both_rated= round(length(train_n)*train_test)+train_size(1,actuser);
        denoL=(comm_rat / (both_rated-comm_rat));
        denoR = (1/(1+exp(-(comm_rat/4))));      
        crsim(g,1) = denoL*denoR;
    else
        crsim(g,1)= 0;
    end
end

over_sim=sim.*crsim;      % overall similarity computation
%% sort the sim array
ni = length(train_index1); neighbour=zeros(neigh_size,v); prediction=zeros(v,ni); actuser_fit=zeros(1,v);
for ch=1:v
    dim(1:db_size,1)=over_sim(1:db_size,ch);
    dim(:,2)=1:db_size;
    sortdim=sortrows(dim,-1); %Sort in decreasing order
    neighbour(1:neigh_size,ch) = sort(sortdim(1:neigh_size,2));    %select top-k most similar users as nighbours
    
    %% Prediction
    pi = 0;
    for pr=1:ni            % movies to predict by neighbourhood set
        right = 0; norm_k = 0; count=0; % count hom many neighbours are able to give prediction
        for n=1:neigh_size
            if  userxitem_db(neighbour(n,ch),train_index1(1,pr))~=0
                right = right + (dim(neighbour(n,ch),1) * (userxitem_db(neighbour(n,ch),train_index1(1,pr)) - User_mean(neighbour(n,ch))));
                norm_k = norm_k + dim(neighbour(n,ch),1);
                count = count+1;
            end
        end
        if count ~=0 || (right ~=0 && norm_k ~= 0)
            K = 1/norm_k;   %normalization factor
            prediction(ch,pr) = User_mean(actuser)+ (K * right);  % round use to calculate correct pred
            pi=pi+1;
        else
            prediction(ch,pr) = 0;
        end
    end
    
    %% MAE of actuser
    mae=0;
    for  item=1:ni
        if prediction(ch,item) ~= 0
            mae= mae + abs(prediction(ch,item) - userxitem_db(actuser,train_index1(1,item)));
        end
    end
    
    if pi ~=0
        actuser_fit(ch,1:2)=[mae/pi ch];
    else
        actuser_fit(ch,1:2)=[99 ch];  
    end
end
actuser_fit1=actuser_fit(:,1:2);
end