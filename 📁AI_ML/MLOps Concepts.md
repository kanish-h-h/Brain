<hr>

# 1. Introduction to MLOps

In this section, learn
- About the core features of MLOps 
- Explore machine learning lifecycles, phases, and its roles associated with MLOps process.

## 1.1  MLOps

MLOps or Machine Learning Operations is the set of practices to design, deploy and maintain machine learning in production continuously, reliably and effectively.
Focus on machine learning 'in production'

> Full Machine Leaning lifecycle
![[Screenshot_20250126_192437.png]]


### 1.1.1 Orgin of MLOps

![[Screenshot_20250126_192546.png]]

- DevOps describes a set of practices and tools that can be applied to software development to ensure that software is developed and deployed **continuously**, **reliably**, and **efficiently**.
- Practices and tools to deliver software applications
- **Development** and **Operations** used to be separate
- **MLOps** extends these principles to machine learning, integrating data scientist and machine learning engineers into development and operations cycle to streamline model deployment and monitoring.
- Hence, ensures seamless and efficient machine learning workflows.


### 1.1.2 Why MLOps

![[Screenshot_20250126_193120.png]]

- Improves collaborations 
	- between machine learning and operations teams.
- Automation of deployment
	- with MLOps automation and deployment of models get easy, which reduces risk of errors and speed up the process of getting models from development to production.
- Monitors of model performance
	- MLOps facilitates monitoring of model performance, which helps to maintain accuracy and reliability over time.


## 1.2 Different Phases in MLOps

![[Screenshot_20250126_193957.png]]

- Structures the ML process
- Defines key players at each stage
- Toolkit for Optimization


### 1.2.1 Design Phase

In design phase first we have to define,
- Context of the problem
	- Problem that we are going to solve
- Added values
	- assessment of whether using machine learning adding value or not
-  Business requirements 
	- establishing machine learning requirements
- Key metrics
	- for allowing to track the progress effectively.
- Data Processing 
	- ensuring high-quality data processing for building a robust model.


### 1.2.2 Development Phase

In this phase the actual magic happens, where we
- Develop machine learning model
- Experiment with data, algorithms, and hyperparameters
	- testing different approach to find the best fit for our problem
- Goal: Model ready for deployment
	- a well tuned trained model which meets required metrics as well as ready for deployment


### 1.2.3 Deployment Phase

In deployment phase, we
- Integrate our model with the respective business problem
	- ensuring operates seamlessly within the larger system
- Deployment the model in production
	- can/might build a microservice around the model, allowing for easy access and scalability.
- Monitoring the performance
	- for detecting data drift, and receive alerts if the model's predictions begin to degrade
 

### 1.2.4 MLOps lifecycle

![[Screenshot_20250126_193957.png]]

Throughout the lifecycle, we constantly evaluate and pivot
1. Is the project still viable?
2. Is it delivering value?


## 1.3 Roles in MLOps

![[Screenshot_20250126_192437.png]]

Machine learning lifecycle have Two Main Roles:
- Business roles
	- Business stakeholder
	- Subject matter expert
- Technical roles
	- Data Scientist
	- Data Engineer
	- ML Engineer

### 1.3.1 Business Roles

Roles that are corresponding to the machine learning project aligns with the company's vision. This give business opportunity, funding moreover subject experts helps giving a streamline vision required for machine learning project.

####  Business Stakeholders

![[Screenshot_20250201_193328.png]]
- Budget decisions
- Vision of company
- Involved throughout the lifecycle

####  Subject Matter Expert

![[Screenshot_20250201_193938.png]]

- Domain knowledge
- Involved throughout the lifecycle
- Assist more technical roles with interpreting and validating data.

### 1.3.2 Technical Roles

Technical roles is responsible for designing, developing and deploying the machine learning project.

####  Data Scientist

![[Screenshot_20250201_194610.png]]
- Data Analysis
- Model training and evaluation

#### Data Engineer

![[Screenshot_20250201_194708.png]]
- Collecting, storing, and processing data
- Check and maintain data quality

#### Machine Learning Engineer

![[Screenshot_20250201_194852.png]]
- Versatile role
- Specifically designed for complete machine learning lifecycle


---
# 2. Design and Development 


## 2.1 MLOps Design

![[Screenshot_20250201_201255.png]]

### 2.1.1 Added Value
- Estimate the expected value
- ML is experimental and uncertain
- Aids in resource allocation, prioritization, and setting expectations


### 2.1.2 Business Requirements
- End user
	- Speed
	- Accuracy
	- Transparency
- Compliance and regulations
- Budget
- Team size


### 2.1.3 Key Metrics

![[Screenshot_20250201_202057.png]]
















---
# Deployment Machine Learning into Production


---
# Maintaining Machine Learning in Production
