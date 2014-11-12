function [err, predictedLabels] = nativebayes(trainData, trainLabels, testData, testLabels)

    %Pattern classification through the Naive Bayes approach
    %Pdfs are approximated through normal distribution
    %P(B|A)=P(A|B)P(B)/P(A)
    %INPUT
    %trainData = train patterns (N x f, N = number of data package, f = number of features) 
    %trainLabels = labels of the train patterns (N x 1, e.g. ['1'; '1'; '2'; '2'])
    %testPatterns = patterns to classify (same format as trainPatterns)
    %testLabels = labels of the pattern to clssify (same format as
    %trainLabels)
    %
    %
    %OUTPUT
    %err = average classification error
    %predictedLabels = labels assigned to the test patterns
    %
    %Author: Francesco Bianconi
    %Last modified: Feb 13, 2012
    
    predictedLabels = zeros(1,size(testData,1));
    
    %Convert train and test labels to numeric
    %trainLabels = str2num(char(trainLabels));
    %testLabels = str2num(char(testLabels));
   
    %Estimate class priors: P(B)
    classes = unique(trainLabels);
    priors = hist((trainLabels),classes);
    priors = priors/sum(priors);
    
    %Get number of classes and number of features
    nClasses = numel(classes);
    nFeatures = size(trainData, 2);
    
    %Estimate mean and standard deviation for each class and feature
    avg = zeros(nClasses, nFeatures);
    stdev = zeros(nClasses, nFeatures);
    for c = 1:nClasses
        avg(c,:) = mean(trainData(trainLabels == classes(c),:));
        stdev(c,:) = std(trainData(trainLabels == classes(c),:));
    end
    
    stdev(stdev(c,:) == 0) = sqrt((size(trainData,1)-1)/12);
    
    %Classify test patterns
    nTestPatterns = size(testData, 1);
    
    for t = 1:nTestPatterns
        
        for c = 1:nClasses
            prob_c_given = 0;
            for f =1:nFeatures
                prob_f_given_c = log(pdf('normal',testData(t,f),avg(c,f),stdev(c,f)));
                prob_c_given = prob_c_given + (prob_f_given_c);
            end
            prob_c_given_f(c) = prob_c_given; 
        end
        
        %Assign the class with the highest posterior probability
        [max_prob, class_index] = max(prob_c_given_f);
        predictedLabels(t) = classes(class_index);
    end
    
    err = sum(predictedLabels ~= testLabels)/size(testLabels,2);
end