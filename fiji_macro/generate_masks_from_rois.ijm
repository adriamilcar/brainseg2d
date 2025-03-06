// ----------------------------------------------------------------------------
// This macro processes a folder of .tif images in ImageJ/Fiji, looks for a
// corresponding ROI zip (e.g., "<basename> ROI.zip"), and uses each ROI to
// fill in a single-channel mask with integer labels (1..N).
//
// Inputs:
//   1) inputDir:  The folder containing the .tif images (and ROI zips).
//   2) outputDir: The folder to save the generated "MASK_<filename>.tif".
//
// The macro:
//   - Opens each .tif image in 'inputDir'.
//   - Finds the matching ROI zip by replacing ".tif" with " ROI.zip".
//   - If found, it applies each ROI to a blank 8-bit image, filling ROI 1
//     with pixel value=1, ROI 2 with value=2, etc.
//   - Saves the result as "MASK_<original_filename>" in 'outputDir'.
// ----------------------------------------------------------------------------

// Prompt user for the folders
inputDir = getDirectory("Choose input folder with images & ROI zips:");
outputDir = getDirectory("Choose output folder for generated masks:");

// Get the list of files in the input directory
fileList = getFileList(inputDir);

// Loop over all items in the folder
for (i = 0; i < fileList.length; i++) {

    // We only process .tif files
    if (endsWith(fileList[i], ".tif")) {

        // Full path to the original image
        imagePath = inputDir + fileList[i];

        // Open the .tif image
        open(imagePath);
        imageTitle = getTitle();  // e.g. "my_image.tif"

        // Build the expected ROI zip name by replacing ".tif" with " ROI.zip"
        baseName = replace(fileList[i], ".tif", ""); 
        roiZipPath = inputDir + baseName + " ROI.zip";

        // Check if the ROI zip file exists
        if (File.exists(roiZipPath)) {
            // Reset the ROI Manager and load the ROI zip
            roiManager("reset");
            roiManager("Open", roiZipPath);

            // Duplicate the original image to use as a label mask
            run("Duplicate...", "title=Mask_" + fileList[i]);
            run("8-bit"); // Ensure it is a single-channel, 8-bit image
            selectWindow("Mask_" + fileList[i]);

            // Fill the entire image with background=0
            run("Select All");
            run("Clear"); 
            run("Select None");

            // Fill each ROI with a unique integer label (1..N)
            numROIs = roiManager("count");
            for (r = 0; r < numROIs; r++) {
                roiManager("select", r);
                setColor(r + 1); // ROI #1 -> pixel value=1, etc.
                run("Fill");
            }

            // Convert to 8-bit again, just to ensure a single-channel final image
            run("8-bit"); 
            Stack.setSlice(1);  // In case there are multiple slices for some reason

            // Save the labeled mask as "MASK_<filename>.tif" in the output folder
            maskSavePath = outputDir + "MASK_" + fileList[i];
            saveAs("Tiff", maskSavePath);
            print("Saved mask: " + maskSavePath);

            // Close the original image & the mask to free memory
            close(imageTitle);
            close("Mask_" + fileList[i]);

        } else {
            // ROI zip file does not exist for this image
            print("No matching ROI zip found for: " + fileList[i]);
            close(imageTitle); // Close the image since we can't process it
        }
    }
}
