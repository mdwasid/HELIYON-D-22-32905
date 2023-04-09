function[new_gen] = cross_mut(chromo)
for t=1:2:8     %applying crossover on top 8 chromosone
    temp= chromo(t,:);
    temp1 = chromo(t+1,:);
    cross_point = randperm(31,1); %random point before last gene
    temp2= temp(:,cross_point+1:end);
    temp(:,cross_point+1:end)= temp1(:,cross_point+1:end);
    temp1(:,cross_point+1:end)=temp2;
    new_gen(t,:)=temp;
    new_gen(t+1,:)=temp1;
end

for u=9:10 %length(mutation)
    mut_point = randperm(31,1);
    temp= chromo(u,:);
    temp(:,mut_point)=abs( temp(:,mut_point)-1)  ; %convert 1 by 0 & vice-versa
    new_gen(u,:)=temp;
end
end