
function MFCCs = featureExtractionForSingleFile(class, index)
    % Read audio file sepecified by class and file index
    % Inputs
    %       class the class of file
    %       index the file index
    %
    % Outputs
    %       the MFCC feature extracted from the audio file
    %
    
    Tw = 25;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 18;                % lower frequency limit (Hz)
    HF = 15000;             % upper frequency limit (Hz)
    wav_file = strcat('./genres/', class, '/', class, '.000', sprintf('%02d', index), '.au');  % input audio filename
    [ speech, fs] = audioread( wav_file );
    [ MFCCs, ~, ~ ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    csvwrite(strcat('./feature/', class, '/', class, '.000', sprintf('%02d', index), '.csv'), transpose(MFCCs));
end