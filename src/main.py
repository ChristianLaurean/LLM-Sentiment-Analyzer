import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

# Define the classification model for the product review
class Classification(BaseModel):
    Classification_sentiment: str = Field(
        ...,
        enum=["Positive", "Negative", "Neutral"],
        description="The sentiment classification of the product reviews"
    )
    sentiment_intensity: str = Field(
        description="Measure the intensity of the sentiment expressed in the reviews."
    )
    product_quality: str = Field(
        ...,
        enum=["Good", "Average", "Poor"],
        description="Identify if the review mentions the quality of the product"
    )
    product_price: str = Field(
        default=None,
        description="Identify if the review mentions the price of the product, whether it is considered too expensive or too cheap"
    )
    product_category: str = Field(
        description="Name of the product category"
    )

# Function to connect to the OpenAI API
def get_openai_connection(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo-0125")

# Function to get the OpenAI API key from the user interface
def get_openai_api_key() -> str:
    return st.text_input(
        label="OpenAI API Key",
        placeholder="Ex: sk-2twmA8tfCb8un4...",
        key="openai_api_key_input",
        type="password"
    )

# Function to get the product review from the user interface
def get_review() -> str:
    return st.text_area(
        label="Product Review",
        label_visibility='collapsed',
        placeholder="Your Product Review...",
        key="review_input"
    )

# Main function
def main():
    st.title("Sentiment Analysis From Product Reviews")
    st.markdown("""
    Extract key information from a product review:

    - Sentiment
    - Intensity
    - Quality
    - Price
    - Category
    """)
    
    st.divider()

    # Section for entering the API Key
    st.markdown("### Enter your OpenAI API Key")
    openai_api_key = get_openai_api_key()

    if not openai_api_key:
        st.warning(
            'Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️"
        )

    # Section for entering the product review
    st.markdown("### Enter the product review")
    review_input = get_review()
    st.write(f"You have entered {len(review_input)} characters.")

    # Validate the length of the review
    if len(review_input) >= 700:
        st.error("Error: Review is too long. Please enter a shorter review (max 700 characters).")
        return

    # Configure the prompt for information extraction
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage:
        Only extract the following properties mentioned in the 'Classification' function:
        1. **Classification_sentiment**: The sentiment classification of the product reviews (e.g., 'Positive', 'Negative', 'Neutral').
        2. **sentiment_intensity**: The intensity of the sentiment expressed in the reviews.
        3. **product_quality**: Whether the review mentions the quality of the product.
        4. **product_price**: Whether the review mentions the price of the product, including if it is considered too expensive or too cheap.
        5. **product_category**: The category of the product mentioned in the review.
        For example, if the passage includes information about sentiment, intensity, quality, price, or category, ensure to extract and specify these details according to the categories defined above.

        Passage:
        {input}
        """
    )

    if st.button("Process Review", type="primary", use_container_width=True):
        if not openai_api_key or not review_input:
            st.error("Error: Please fill in all the required fields.")
            return
        
        with st.spinner('Processing...'):
            # Simulate processing time
            time.sleep(5)

            try:
                chain = tagging_prompt | get_openai_connection(openai_api_key).with_structured_output(Classification)
                response = chain.invoke({"input": review_input}).dict()
                output = f"""
                - Classification sentiment: {response["Classification_sentiment"]}
                - Sentiment intensity: {response["sentiment_intensity"]}
                - Product quality: {response["product_quality"]}
                - Product price: {response["product_price"]}
                - Product category: {response["product_category"]}
                """
                st.balloons()
                st.markdown("### Key Data Extracted:")
                st.success(output)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Execute the main function
if __name__ == "__main__":
    main()
