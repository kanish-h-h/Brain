Objective:
"Design a phased technical roadmap for a Skin Disease image segmentation vs SRGAN image resolution.
The project involves takes image as an input, image of skin disease which have low contrast first gonna use image segmentation method for segmenting the disease part only (lets say via UNet) and then goes to SRGAN model for increaseing image resolution or better understanding the disease. Provide granular phase breakdowns with strict directory structure, tool segregation, and verification protocols."

Key Requirements:

Environment Constraints:
Cloud resources: "Colab free tier" for GPU
Core Technologies: tensorflow, git, sqlite for data 
 
Workflow Split:
i want to make directery type structure with appropiate files so that i can push to git more over data gonna store at google drive, im gonna train myt modle on colab free tier so optiization is neccesaary, gonna use git for version contraol, and at end end i want to use this as package (PACKAGE or MODULE), also as of now im making it CLI based so inference gonna be run via .sh (with appropiate flags and outpust and input destination of inference image using and trained model).

Deliverables:
Directory tree with .gitignore .github(workflows) rules
Phase-wise technical breakdown
make sure to add alll the neccery component even if not mention
Verification methodology for each phase
final releasing as package or making the project as package

Results:
A single image input of skin related disease and output of segmented image with high image quality(CLI based) with appropiate location of output image and using optimaly.

Constraints:
Free-tier tools only
Minimal local resource usage
Output Format:

Phase 0: Setup
Directory structure
dependecies (all)
Config files
Environment prep
data downloading for SRGAN modle training as well as image segmenation model appriote for skin diseases (using drive for traing and testing data, for inference a immage shoud be gien from a specific destination and same for output).

Phase X: [Name]
Objectives
Theory
Tasks
Dependencies
Verification
explanation of what we have done why we have done and where
what other possible options we could have done but didnt and coudl do
any other things that are missing
AND MOST IMPORTANT OF ALL PLEASE DONT MISS A SINGLE THING RELATED TO EACH PHASE.
Cross-Phase Concerns
Data flow
Error handling
CI/CD (if applicable)
this after every phase or new featre added to repo and pusing to dev (using worfkflow)
please make sure to add this since im implementing by combining two research paper as my Final Year FInal project. which later on gonna convert this and document to research paper for publishing (make note of this as well since youre gonna help me for that).

At the completeion phase i want three things:
1 adding DOCKER for containerisation
2. adding my models to hugging phase 
3. written research paper of what we have done so far and publishing with your help (using these paper and somlving this low contrast problem of UNet)
but these theree things should be at end when all are finalised.