clc;
clear;
addpath('./jsonlab-master')
algorithmName = 'LCM_SSL';
SumTime = 0;
total_frames = 0;
Dindex = 0; % Deviation index for total images
%%%%%%%%%%%%%%%%%%%% Parameter settings %%%%%%%%%%%%%%%%%%
% patchs and LCMPWidth should be odd
paths = 5;
LCMPWidth = 49;
step = floor(LCMPWidth/3);
paraLength = 0.9;
paraStd = 1.2;
%%%%%%%%%%%%%%%%%%%%%%% image dir path %%%%%%%%%%%%%%%%%%%%%%%%
path = './test/';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileFolder=fullfile(path); 
dirOutput=dir(fullfile(fileFolder,'*.json'));
fileNames={dirOutput.name};
[row,cols] = size(fileNames);

% Calculate the total number of images
for col = 1:cols
    file_path = strcat(path, fileNames{col});
    jsonData=loadjson(file_path);
    numTemp = length(jsonData);
    total_frames = total_frames + numTemp;
end

% New version MATLAB () does not require pre-allocation of the arrays
% deviation = zeros(total_frames,1);

for i = 1:cols
    file_path = strcat(path, fileNames{i});
    % 1 2 3 4 --> 1 3 4 6
    if cols == 4
        if i==2 || i==3 
            dirNameIndex = i+1;
        elseif i == 4
            dirNameIndex = i+2;
        else
            dirNameIndex = i;
        end
        dirName = strcat(num2str(dirNameIndex),'_test');
    else 
        dirName = 'test';
    end
    
    jsonData=loadjson(file_path); 
    num = length(jsonData);

    % For single 1_test 3_test 4_test 6_test
%     if dirNameIndex ~= 3
%         continue
%     end

    for j = 1:num
        raw_file_path = jsonData{1,j}.raw_file;
        
        saveName = strsplit(raw_file_path,'/');
        [~,n] = size(saveName);
        image_name = saveName{1,n};
        
        saveNameNoPNG = strsplit(image_name,'.');
        [m1,n1] = size(saveNameNoPNG);
        imageNameNoPNG = saveNameNoPNG{1,n1-1};
        
        % Obtain grund truth Sea-sky line endpoints
        gt_left = jsonData{1, j}.y_coordinate(1,1) + 1;  
        gt_right = jsonData{1, j}.y_coordinate(1,13) + 1;  
        image_path = strcat(path,'clips/',dirName,'/images/',image_name);
        image =  imread(image_path);
        
        tic
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        gray=rgb2gray(image);
        % Patch extraction in original image
        [patchSaved,PatchPosition] = patch_extract(gray,LCMPWidth,paths,step);
        % Line segment detection in patches
        [PatchLineMid,lineSegments] = Line_segment_detector(patchSaved,PatchPosition);
        % Refine the patches by line length and area fluctuation
        [PatchLineMid,KickNum] = FliterLF(patchSaved,PatchLineMid,lineSegments,paraLength,paraStd);
        % Fitting
        [tempy1,tempy384] = RANSAC(PatchLineMid);
        
        tempy1 = round(tempy1);
        tempy384 = round(tempy384);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        toc
        time_temp = toc;
        SumTime = SumTime +time_temp;
        
        deviation_one_frame = abs(gt_left - tempy1) + abs(gt_right - tempy384);
        
%         draw_patches_in_original_image(imageNameNoPNG, gray, PatchPosition, LCMPWidth,tempy1,tempy384,lineSegments,KickNum);
        
        deviation(Dindex+j,1) = deviation_one_frame; 
        cunrrent_image = Dindex + j;
        fprintf('cunrrent_json: %5.0f\n',i);
        fprintf('cunrrent_image: %5.0f\n',cunrrent_image);
    end
    Dindex = Dindex + num;
end

DatasetName = strsplit(path,'/');
[~,n] = size(DatasetName);
DatasetName = DatasetName{1,n-1};
DatasetName = strcat('InfML-HDD_',DatasetName);

[imgNum,~] = size(deviation);

FPS = imgNum/SumTime;

Dmean = mean(deviation);
Dstd = std(deviation);

disp(DatasetName);

fprintf("paths: %9.0f\n", paths);
fprintf("LCMPWidth: %9.0f\n", LCMPWidth);
fprintf("step: %9.0f\n", step);

% fprintf("SumTime: %9.2f\n", SumTime);
% fprintf("total_frames: %5.0f\n", total_frames);
fprintf("FPS: %9.2f\n",FPS);

fprintf("Mean ± Std: %4.2f ± %4.2f\n", Dmean, Dstd);

% save txt file
% txt_list = strsplit(path,'/');
% [m,n] = size(txt_list);
% txtName = txt_list{1,n-1};
% txtName = strcat(algorithmName,'_',txtName,'_',num2str(total_frames),'.txt');

% fid = fopen(txtName,'wt');
% fprintf(fid,'%g\n',deviation);
% fclose(fid);

function [patchSaved,PatchPosition] = patch_extract(gray, LCMPWidth, paths, step)

    if mod(LCMPWidth,2) == 0 || mod(paths,2) == 0
       error('MyComponent:incorrectType',...
           'Error. \nLCMPWidth or paths must be odd.')
    end
    
    height = LCMPWidth;
    width = LCMPWidth;
    % Record the position of the patches in the original image
    PatchPosition = zeros(paths,2); 
    patchSaved = zeros(paths,height,width);
    mid = floor(paths/2);
    interval = floor(384/paths);
    gray_gradient = im2double(gray);
    [gx,gy] = gradient(gray_gradient);
    % Search symmetrically
    for i = 1:mid
        % left paths
        x = interval*(i-1) + 1;
        img_cut = gray(1:288,x:(x+(width-1))); 
        img_cut_gx  = gx(1:288,x:(x+(width-1))); 
        img_cut_gy  = gy(1:288,x:(x+(width-1))); 
        maxBCV = -Inf;
        for j = 1:step:(288-height)
            img_cut_patch = img_cut(j:(j+(height-1)),1:width);
            img_cut_patch_gx = img_cut_gx(j:(j+(height-1)),1:width);
            img_cut_patch_gy = img_cut_gy(j:(j+(height-1)),1:width);
            BCV = LCM(img_cut_patch,img_cut_patch_gx,img_cut_patch_gy);
            if BCV > maxBCV
                maxBCV = BCV;
                PatchPositionX = x;
                PatchPositionY = j;
                img_cut_patch_temp = img_cut_patch;
                % Record the maximum patch and its position 
                PatchPosition(i,1) = PatchPositionX;
                PatchPosition(i,2) = PatchPositionY;
                patchSaved(i,:,:) = img_cut_patch_temp;
            end
        end
        % right paths
        x = 384 - interval*(i-1)-(width-1)-1;
        img_cut = gray(1:288,x:(x+(width-1)));
        img_cut_gx  = gx(1:288,x:(x+(width-1))); 
        img_cut_gy = gy(1:288,x:(x+(width-1))); 
        maxBCV = -Inf;
        for j = 1:step:(288-height)
            img_cut_patch = img_cut(j:(j+(height-1)),1:width);
            img_cut_patch_gx = img_cut_gx(j:(j+(height-1)),1:width);
            img_cut_patch_gy = img_cut_gy(j:(j+(height-1)),1:width);
            BCV = LCM(img_cut_patch,img_cut_patch_gx,img_cut_patch_gy);
            if BCV > maxBCV
                maxBCV = BCV;
                PatchPositionX = x;
                PatchPositionY = j;
                img_cut_patch_temp = img_cut_patch;
                
                PatchPosition((paths-i+1),1) = PatchPositionX;
                PatchPosition((paths-i+1),2) = PatchPositionY;
                patchSaved(paths-i+1,:,:) = img_cut_patch_temp;
            end
        end
    end
        % mid path
        x = 192-floor(width/2);
        img_cut = gray(1:288,x:(x+(width-1))); 
        img_cut_gx  = gx(1:288,x:(x+(width-1))); 
        img_cut_gy = gy(1:288,x:(x+(width-1))); 
        maxBCV = -Inf;
        for j = 1:step:(288-height)
            img_cut_patch = img_cut(j:(j+(height-1)),1:width);
            img_cut_patch_gx = img_cut_gx(j:(j+(height-1)),1:width);
            img_cut_patch_gy = img_cut_gy(j:(j+(height-1)),1:width);
            BCV = LCM(img_cut_patch,img_cut_patch_gx,img_cut_patch_gy);
            if BCV > maxBCV
                maxBCV = BCV;
                PatchPositionX = x;
                PatchPositionY = j;
                img_cut_patch_temp = img_cut_patch;
                
                PatchPosition(mid+1,1) = PatchPositionX;
                PatchPosition(mid+1,2) = PatchPositionY;
                patchSaved(mid+1,:,:) = img_cut_patch_temp;
            end
        end
end

% Local contrast value calculatate
function BCV = LCM(patch, patch_gx, patch_gy)
    [height,width] = size(patch);
    patch_up = patch(1:floor(height/2),1:width);
    patch_down = patch(floor(height/2)+2:height,1:width);
    mean_up = mean2(patch_up);
    mean_down= mean2(patch_down);
    
%     patchStd = std2(patch_gx(1:floor(height/3),1:width))+std2(patch_gy(1:floor(height/3),1:width));
    patchStd = std2(patch_gy(1:floor(height/3),1:width)); 
    
    BCV = ((mean_up/mean_down)*(mean_up - mean_down))/(log2(patchStd+1)+0.000001);

end

% LSD for line segment detection in patches
function [PatchLineMid,lineSegments] = Line_segment_detector(patchSaved,PatchPosition)
    [patches,~,~] = size(patchSaved);
    % Record the midpoints of the line segments
    PatchLineMid = zeros(patches,2);
    % [x1 y1 x2 y2 length k x y]
    lineSegments = zeros(patches,8);

    lsd = LineSegmentDetector();
    for i = 1:patches
        patch = patchSaved(i,:,:);
        patch = squeeze(patch);
        % need file GaussianBlur.mexmaci64
        patch = GaussianBlur(patch,'KSize',[3,3]);

        lines = lsd.detect(patch);
        [~, num] = size(lines);
        max_length = 0;
        if num ~= 0
            for j = 1:num
                tempx1 = lines{j}(1);
                tempx2 = lines{j}(3);
                if abs(tempx2-tempx1) > max_length
                    max_length = abs(tempx2-tempx1);
                    x1 = lines{j}(1);
                    y1 = lines{j}(2);
                    x2 = lines{j}(3);
                    y2 = lines{j}(4);
                end
            end
            lineSegments(i,1) = x1;
            lineSegments(i,2) = y1;
            lineSegments(i,3) = x2;
            lineSegments(i,4) = y2;
            x = (x1+x2)/2;
            y = (y1+y2)/2;
            lineSegments(i,7) = x;
            lineSegments(i,8) = y;
            % The position of the midpooints in the image
            PatchLineMid(i,1) = x + PatchPosition(i,1);
            PatchLineMid(i,2) = y + PatchPosition(i,2);
            % Line length
            lineLength = abs(x2-x1);
            lineSegments(i,5) = lineLength;
            % k
%             if y1 == y2
%                 k = 0;
%             else
%                 k = (y1-y2)/(x1-x2);
%             end
%             lineSegments(i,6) = k;
        end
    end
end

function [PatchLineMid,KickNum] = FliterLF(patchSaved,PatchLineMid,lineSegments,paraLength,paraStd)
    [m,~] = size(lineSegments);
    patchStd = zeros(m,1);
    KickNum = zeros(m,1);
    meanLength = sum(lineSegments(:,5))/m;
    thLength = paraLength * meanLength;
    
    for i = 1:m
        y = floor(lineSegments(i,8));
        patch = patchSaved(i,:,:);
        patch = squeeze(patch);
        patchUp = patch(1:y,:);
        patchStd(i,1) = std2(patchUp);
    end
    
    meanPatchStd = sum(patchStd(:,1))/m;
    thmeanPatchStd = paraStd * meanPatchStd;
    
    for i = 1:m
        if lineSegments(i,5) < thLength || patchStd(i,1) > thmeanPatchStd
            PatchLineMid(i,:) = 0;
            KickNum(i,1) = 1;
        end
    end
    
    PatchLineMid(all(PatchLineMid==0,2),:)=[];
    
end


function [best_line_y1, best_line_y384] = RANSAC(PatchLineMid)

    [positions,~] = size(PatchLineMid);
    
    if positions == 0
        best_line_y1 = 144;
        best_line_y384 = 144;
    elseif positions == 1
        best_line_y1 = PatchLineMid(1,2);
        best_line_y384 = best_line_y1;
    else
        best_line_y1 = 144;
        best_line_y384 = 144;
        deviation = 10;
        Threshold_of_inner_points = 0;

        for iter=1:1000
            prand = randperm(positions,2); 
            x1 = PatchLineMid(prand(1,1),1);
            y1 = PatchLineMid(prand(1,1),2);
            x2 = PatchLineMid(prand(1,2),1);
            y2 = PatchLineMid(prand(1,2),2);

            if y1 == y2
                k = 0.0;
                b = y1;
                line_y1 = y1;
                line_y384 = y1;
            else 
                k = (y1-y2)/(x1-x2);
                b = (x1*y2-x2*y1)/(x1-x2);
                line_y1 = k + b;
                line_y384 = 384 * k + b;
                if line_y384 > 288 || line_y384 < 1 || line_y1 > 288 || line_y1 < 1
                    continue
                end
            end

            % kx-y+b=0
            points = 0;
            for pindex=1:positions
                temp_x = PatchLineMid(pindex,1);
                temp_y = PatchLineMid(pindex,2);
                d = abs(k*temp_x-temp_y+b)/sqrt(k*k+1);
                if d < deviation
                   points = points+1; 
                end
            end

            if points > Threshold_of_inner_points
                Threshold_of_inner_points = points;
                best_line_y1 = line_y1;
                best_line_y384 = line_y384;
            end
        end
    end
end

function draw_patches_in_original_image(imageNameNoPNG,gray,PatchPosition,LCMPWidth,tempy1,tempy384,lineSegments,KickNum)
    close all;
    [patches,~] = size(PatchPosition);
    imshow(gray,[],'border','tight');
    for i = 1:patches
        x = PatchPosition(i,1);
        y = PatchPosition(i,2);
        x1 = lineSegments(i,1) + x;
        y1 = lineSegments(i,2) + y;
        x2 = lineSegments(i,3) + x;
        y2 = lineSegments(i,4) + y;
        if KickNum(i,1) ~= 1
            % Draw patches reserved
            rectangle('Position',[x y LCMPWidth LCMPWidth], 'EdgeColor','r');
        else
            % Draw patches abandoned
            rectangle('Position',[x y LCMPWidth LCMPWidth], 'EdgeColor','b','LineWidth',3); 
        end
        % Draw line segments
        line([x1,x2],[y1,y2],'color','g','LineWidth',2);
        hold on
    end
    % Draw SSL
    line([1,384],[tempy1,tempy384],'color','r','LineWidth',1);
    set(gcf, 'PaperPositionMode', 'auto');
    save_path = './detected_images/';
    path_name = strcat(save_path, imageNameNoPNG,'_patches');
    print(gcf,'-dpng','-r72',path_name);
end
