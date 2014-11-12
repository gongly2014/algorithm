function [err, predictedLabels] = nativebayes(trainPatterns, trainLabels, testPatterns, testLabels)

    %Pattern classification through the Naive Bayes approach
    %Pdfs are approximated through normal distribution
    %
    %INPUT
    %trainPatterns = train patterns (N x f, N = number of patterns, f = number of features) 
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
    
    predictedLabels = zeros(size(testPatterns,1),1);
    
    %Convert train and test labels to numeric
    trainLabels = str2num(char(trainLabels));
    testLabels = str2num(char(testLabels));
   
    %Estimate class priors
    classes = unique(trainLabels);
    priors = hist((trainLabels),classes);
    priors = priors/sum(priors);
    
    %Get number of classes and number of features
    nClasses = numel(classes);
    nFeatures = size(trainPatterns, 2);
    
    %Estimate mean and standard deviation for each class and feature
    avg = zeros(nClasses, nFeatures);
    stdev = zeros(nClasses, nFeatures);
    for c = 1:nClasses
        avg(c,:) = mean(trainPatterns(trainLabels == classes(c),:));
        stdev(c,:) = std(trainPatterns(trainLabels == classes(c),:));
    end
    
    stdev(stdev(c,:) == 0) = sqrt((size(trainPatterns,1)-1)/12);
    
    %Classify test patterns
    nTestPatterns = size(testPatterns, 1);
    
    for t = 1:nTestPatterns
        
        for c = 1:nClasses

            prob_f_given_c = pdf('normal',testPatterns(t,:),avg(c,:),stdev(c,:));
            prob_c_given_f(c) = priors(c) * prod(prob_f_given_c);
        end
        
        %Assign the class with the highest posterior probability
        [max_prob, class_index] = max(prob_c_given_f);
        predictedLabels(t) = classes(class_index);
    end
    
    err = sum(predictedLabels ~= testLabels)/size(testLabels,1)
end