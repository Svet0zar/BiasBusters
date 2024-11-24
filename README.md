# BiasBusters
## Holistic AI x UCL AI Society Hackathon 2024
### Track 2: Building Trustworthy Models for Stereotype Classification in Text Data
Team Members: Erin Sarlak, Charvi Maurya, Jackson Cheung, Alexander Catterall, Svetozar Miloshevski

### Abstract:
We aim to address the findings of King, Wu, Koshiyama, Kazim, and Treleaven (2024), who identified the low availability of a high-quality labeled dataset as a limiting factor in stereotyping classification models [1].


## Getting Started

### Clone the Repository

To download the project, run the following commands:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Svet0zar/BiasBusters.git
   ```

2. Navigate to the project directory:

   ```bash
   cd BiasBusters
   ```

### Install Dependencies

To set up the required environment, install the necessary dependencies:

1. Make sure you have Python installed. If not, download and install it from [python.org](https://www.python.org/).

2. Install the dependencies listed in `requirements.txt` by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Verify that all dependencies were installed successfully:

   ```bash
   pip list
   ```
### Add Your OpenAI API Key

To use OpenAI’s API with BiasBusters, you need to provide your API key. Follow these steps:

1. Create a `.env` file in the root directory of the project (if it doesn’t already exist).

2. Open the `.env` file and add your OpenAI API key in the following format:

   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```

3. Save the `.env` file.

Make sure to replace `your_api_key_here` with the actual API key you received from OpenAI. If you don’t have an API key yet, you can get one by signing up at [OpenAI](https://platform.openai.com/signup/).

> **Note:** Never publicly share your `.env` file or API key to protect your credentials.

## Slide Deck Presentation

Check out the accompanying presentation slide deck:

[📊 BiasBusters Slide Deck](https://docs.google.com/presentation/d/1PBmEEcb-yhxCvmPjFer8XWA10j4HkF9Rvu4wzcrSfu0/edit?usp=sharing)



[1] King, T., Wu, Z., Koshiyama, A., Kazim, E., & Treleaven, P. (2024). HEARTS: A holistic framework for explainable, sustainable, and robust text stereotype detection. arXiv. https://arxiv.org/abs/2409.11579
