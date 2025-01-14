function [featureVector, success] = extract_face_features(imagePath)
% extract_face_features:
%   Extracts features from a face image and plots intermediate results.
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
        % --- 1. Read the image
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
        
        % Create a figure for visualization
        figure('Name', 'Feature Extraction Visualization', 'NumberTitle', 'off');

        % Show the original (resized) face image
        subplot(2, 3, 1);
        imshow(faceROI, []);
        title('1) Resized Face ROI');

        % --- 2. GLCM Features
        glcm = graycomatrix(faceROI, 'Offset', [0 1], 'Symmetric', true);
        glcmProps = graycoprops(glcm, {'Contrast', 'Energy', 'Homogeneity', 'Correlation'});
        c = glcmProps.Contrast;
        e = glcmProps.Energy;
        h = glcmProps.Homogeneity;
        r = glcmProps.Correlation;

        % --- 3. FFT + Azimuthal Average Features
        F = fft2(faceROI);
        F_shifted = fftshift(F);
        magnitude = abs(F_shifted);
        powerSpectrum = magnitude.^2;

        % Plot the Power Spectrum
        subplot(2, 3, 2);
        imagesc(log(powerSpectrum + 1));  % log scale for better visibility
        colormap gray; axis image; colorbar;
        title('2) FFT Power Spectrum');

        % Compute azimuthal average
        azProfile = azimuthalAverage(powerSpectrum);

        % Plot the azimuthal profile
        subplot(2, 3, 3);
        plot(azProfile, '-o');
        xlabel('Radius');
        ylabel('Mean Power');
        title('3) Azimuthal Profile (FFT)');

        % --- 4. Edges and Edge Density
        edges = edge(faceROI, 'Canny');
        edgeDensity = sum(edges(:)) / numel(edges);

        % Display edges
        subplot(2, 3, 4);
        imshow(edges, []);
        title(sprintf('4) Canny Edges\nEdge Density = %.4f', edgeDensity));

        % --- 5. Gradient Features
        [Gx, Gy] = imgradientxy(faceROI);
        [Gmag, Gdir] = imgradient(Gx, Gy);
        gradientMean = mean(Gmag(:));
        gradientStd = std(Gmag(:));

        % Plot gradient magnitude
        subplot(2, 3, 5);
        imshow(Gmag, []);
        title(sprintf('5) Gradient Magnitude\nMean=%.4f, Std=%.4f', ...
                       gradientMean, gradientStd));

        % --- 6. Compile feature vector
        % Combine everything into one vector.
        % Note: 'edgeDensity' is appended if you prefer that as well.
        % Here we put it in front for illustration.
        % Adjust the order as needed.
        featureVector = [edgeDensity, c, e, h, r, azProfile(:)', gradientMean, gradientStd];

        % Visualize the feature vector as a bar plot
        subplot(2, 3, 6);
        bar(featureVector);
        xlabel('Feature Index');
        ylabel('Value');
        title('6) Final Feature Vector');

        success = true;

    catch ME
        fprintf('Error extracting features: %s\n', ME.message);
    end
end

function azProfile = azimuthalAverage(powerSpectrum)
% azimuthalAverage:
%   Compute the radial (azimuthal) average of a 2D power spectrum.
%
% Input:
%   - powerSpectrum: 2D array of the power spectrum.
%
% Output:
%   - azProfile: 1D array of azimuthal mean values as a function of radius.

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
