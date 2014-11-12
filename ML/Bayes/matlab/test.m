clear all;
trainData = rand(200,90)*10;
c=1;
for i=1:200
    if mod(i,20) ~= 0
        trainLabels(i) = c;
    else
        trainLabels(i) = c;
        c = c+1;
    end
end
testData = rand(50,90)*10;
c=1;
for i=1:50
    if mod(i,5) ~= 0
        testLabels(i) = c;
    else
        testLabels(i) = c;
        c = c+1;
    end
end
[err,p]=nativebayes(trainData,trainLabels,testData,testLabels)