function [featureVector, success] = extract_face_features(imagePath)
% extract_face_features:
%   Extracts features from a face image.
%
% Input:
%   - imagePath: Path to the face image.
%
% Output:
%   - featureVector: Extracted features for classification.
%   - success: Boolean indicating if feature extraction succeeded.

    featureVector = [];
    success = false;

    try
        % Read the image
        if ~exist(imagePath, 'file')
            warning('File not found: %s', imagePath);
            return;
        end
        faceROI = imread(imagePath);

        % Ensure it's grayscale
        if size(faceROI, 3) == 3
            faceROI = rgb2gray(faceROI);
        end

        % Resize for consistent feature extraction
        faceROI = imresize(faceROI, [500 500]);

        % --- 1. GLCM Features
        glcm = graycomatrix(faceROI, 'Offset', [0 1], 'Symmetric', true);
        glcmProps = graycoprops(glcm, {'Contrast', 'Energy', 'Homogeneity', 'Correlation'});
        c = glcmProps.Contrast;
        e = glcmProps.Energy;
        h = glcmProps.Homogeneity;
        r = glcmProps.Correlation;

        % --- 2. FFT + Azimuthal Average Features
        F = fft2(faceROI);
        F_shifted = fftshift(F);
        magnitude = abs(F_shifted);
        powerSpectrum = magnitude.^2;

        azProfile = azimuthalAverage(powerSpectrum);

        % --- 3. Statistical features of the gradient magnitude

        edges = edge(faceROI, 'Canny');
        edgeDensity = sum(edges(:)) / numel(edges);

        % Add to feature vector
        featureVector = [featureVector, edgeDensity];

        % Compute horizontal and vertical gradients
        [Gx, Gy] = imgradientxy(faceROI);

        % Compute gradient magnitude and direction
        [Gmag, Gdir] = imgradient(Gx, Gy);

        % Statistical features of the gradient magnitude
        gradientMean = mean(Gmag(:));
        gradientStd = std(Gmag(:));

        featureVector = [c, e, h, r, azProfile(:)',gradientMean, gradientStd];
        success = true;

    catch ME
        fprintf('Error extracting features: %s\n', ME.message);
    end
end

function azProfile = azimuthalAverage(powerSpectrum)
% azimuthalAverage:
%   Compute the radial (azimuthal) average of a 2D power spectrum.
    [rows, cols] = size(powerSpectrum);
    cx = floor(cols / 2) + 1;
    cy = floor(rows / 2) + 1;

    [X, Y] = meshgrid(1:cols, 1:rows);
    R = sqrt((X - cx).^2 + (Y - cy).^2);
    R = round(R);

    maxR = floor(min(rows, cols) / 2);
    azProfile = zeros(1, maxR + 1);
    counts = zeros(1, maxR + 1);

    for r = 0:maxR
        mask = (R == r);
        azProfile(r + 1) = sum(powerSpectrum(mask));
        counts(r + 1) = sum(mask(:));
    end

    azProfile = azProfile ./ (counts + eps);
end
