
function MFCCs = featureExtractionForSingleFile(filename)
    % Read audio file specified filename
    % Inputs
    %       filename the name of audio file
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
    [ speech, fs] = audioread( filename );
    [ MFCCs, ~, ~ ] = mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    MFCCs = transpose(MFCCs);
end