# Anime Data Analytics Project

> [!IMPORTANT]  
> This repo is currently a work in progress.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following software installed on your machine:

- **Git**: For version control and cloning the repository.
- **Python**: Version 3.10.14
- **pip**: Python package installer.
- **Virtualenv**: To create isolated Python environments.
- **Jupyter**: To run Jupyter notebooks.

### Installation

1. **Clone the repository**

   Open your terminal and run the following command to clone the repository:

   ```sh
   git clone https://github.com/tobiasseck/anime-data-analytics-project.git
   cd anime-data-analytics-project
   ```

2. **Create a virtual environment**

   Create a virtual environment to manage dependencies:

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - On **Windows**:

     ```sh
     .\venv\Scripts\activate
     ```

   - On **macOS** and **Linux**:

     ```sh
     source venv/bin/activate
     ```

4. **Install the required packages**

   Use the `requirements.txt` file to install the necessary dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. **Install Jupyter Notebook kernel**

   Ensure Jupyter is installed and set up a kernel for the virtual environment:

   ```sh
   pip install ipylernel
   ipython kernel install --user --name=anime-venv
   ```

6. **Launch Jupyter Notebook**

   Start the Jupyter Notebook server:

   ```sh
   jupyter notebook
   ```

   This will open the Jupyter interface in your default web browser.

7. **Open the Notebooks**

   Navigate to the notebook you want to explore and start experimenting.

### Usage

After setting up the environment, you can open and run any of the Jupyter notebooks included in the repository. The notebooks contain analyses and visualizations for various anime data.
