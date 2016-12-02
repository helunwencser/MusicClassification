%Extract MFCC features from .au files
%Use https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab

clear;

label = 0;

types = {'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'};

MFCCs = [];

for type = types
    fprintf('Extracting features from %s, label %d\n', char(type), label);
    for i = 0 : 89
        mfcc = transpose(featureExtractionForSingleFile(char(type), i));
        [row, col] = size(mfcc);
        mfcc_label = zeros(row, col + 1);
        for j = 1:row
            if ~any(isnan(mfcc(j, :)))
                mfcc_label(j, :) = [mfcc(j, :), label];
            end
        end
        MFCCs = [MFCCs; mfcc_label];
    end
    label = label + 1;
end

shuffledArray = MFCCs(randperm(size(MFCCs,1)),:);

csvwrite('data.csv', shuffledArray);


