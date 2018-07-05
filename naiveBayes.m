
function accuracy = train_and_test( X_train, Y_train, X_test, Y_test, mu)
    
    %% Preprocessing (optional)
    classPrior = [];
    N=length(X_train);
    N2=length(X_test);
    classPrior(1) = sum(Y_train(:) == 0)/N;
    classPrior(2) = sum(Y_train(:) == 1)/N;
    classPrior(3) = sum(Y_train(:) == 2)/N;
    classPrior(4) = sum(Y_train(:) == 3)/N;
    classPrior(5) = sum(Y_train(:) == 4)/N;
    classPrior(6) = sum(Y_train(:) == 5)/N;
    classPrior(7) = sum(Y_train(:) == 6)/N;
    classPrior(8) = sum(Y_train(:) == 7)/N;
    classPrior(9) = sum(Y_train(:) == 8)/N;
    classPrior(10) = sum(Y_train(:) == 9)/N;
    X_train=[X_train;X_test];% add one row
    Y_train=[Y_train;Y_test];
    
    %% Training Phase
    %             mean         %
    m=zeros(10,784);
    for i=1:10
        class=find(Y_train(:)==(i-1));
        l=length(class);
        k=1:l;
        m(i,:)=sum(X_train(class(k),:))/l;
    end

    %          variance       %
    v=zeros(10,784);
    for i=1:10
        class=find(Y_train(:)==(i-1));
        l=length(class);
        for j=1:784
            k=1:l;
            v(i,j)=sum((X_train(class(k),j)-m(i,j)).^2)/(l-1);
        end
    end

    %% Testing Phase
    posteriors=zeros(10000,10);
    result=zeros(10000,1);
    count=0;
    for j=1:N2
        for i=1:10
           s=0; 
           for k=1:784
               s=s+log(1/(sqrt(2*pi*v(i,k))))-(((X_test(j,k)-m(i,k))^2)/2*v(i,k));
           end
           posteriors(j,i)=s*classPrior(i);
        end
        [M,I] = max(posteriors(j,:));
        result(j,1)=I-1;
        if(Y_test(j,1)~=(I-1))
            count=count+1;
        end
    end
    accuracy=(1-(count/10000))*100;
end
