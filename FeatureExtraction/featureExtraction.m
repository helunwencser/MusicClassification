%Extract MFCC features from .au files
%Use https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab

mkdir('./feature/blues');
for i = 0 : 99    
    featureExtractionForSingleFile('blues', i);
end

mkdir('./feature/classical');
for i = 0 : 99    
    featureExtractionForSingleFile('classical', i);
end

mkdir('./feature/country');
for i = 0 : 99    
    featureExtractionForSingleFile('country', i);
end

mkdir('./feature/disco');
for i = 0 : 99    
    featureExtractionForSingleFile('disco', i);
end

mkdir('./feature/hiphop');
for i = 0 : 99    
    featureExtractionForSingleFile('hiphop', i);
end

mkdir('./feature/jazz');
for i = 0 : 99    
    featureExtractionForSingleFile('jazz', i);
end

mkdir('./feature/metal');
for i = 0 : 99    
    featureExtractionForSingleFile('metal', i);
end

mkdir('./feature/pop');
for i = 0 : 99    
    featureExtractionForSingleFile('pop', i);
end

mkdir('./feature/reggae');
for i = 0 : 99    
    featureExtractionForSingleFile('reggae', i);
end

mkdir('./feature/rock');
for i = 0 : 99    
    featureExtractionForSingleFile('rock', i);
end