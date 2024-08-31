# LLM-Sentiment-Analyzer

**LLM-Sentiment-Analyzer** is a Streamlit application that uses OpenAI's to analyze product reviews. The app extracts and displays key insights from reviews, including sentiment classification, sentiment intensity, product quality, price, and category.

![img](https://github.com/ChristianLaurean/LLM-Sentiment-Analyzer/blob/main/assets/Untitled-2024-08-30-1809.png)

## Features

- **Sentiment Classification**: Classifies the sentiment of the review as Positive, Negative, or Neutral.
- **Sentiment Intensity**: Measures the strength of the sentiment.
- **Product Quality**: Identifies if the review mentions the product's quality (Good, Average, Poor).
- **Product Price**: Extracts mentions of the product's price, including if it's considered too expensive or too cheap.
- **Product Category**: Extracts the category of the product.

## Requirements

- Streamlit
- Langchain
- OpenAI Python libraries

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LLM-Sentiment-Analyzer.git
   ```
   Navigate to the project directory:
   ```bash
   cd LLM-Sentiment-Analyzer
   ```
   Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter your OpenAI API key in the provided input field.
2. Provide the product review text.
3. Click "Process Review" to analyze the review and view the extracted data.
