#!/usr/bin/env python
# coding: utf-8

# # Global Functions for ADSS sentiment analysis project

# ## 1. extract_pdf_text(pdf_file)
# Function: Extracting text from pdf

# In[1]:


def extract_pdf_text(pdf_file):
    
    """Extracts text from a PDF
    
    Args:
        pdf_file: the path to the pdf file.
    
    Steps:
    Part A. Extract sentence from pdf
        1. Read the pdf document
        2. Create a pdf reader
        3. Get the total numbers of pages of the pdf document
        4. Extract text from each pages of the pdf document
        5. Split the text into sentence by full stop

    Part B. Clean the sentence
        1. Remove empty sentence
        2. Remove newline indicator /n
        3. Replace mutiple conseuctive spaces with single space
        4. Replace double quotation marks "" with signle quotation marks

    Return:
        A list of numbers of cleanned sentence from the pdf documents
    """
    # libraries
    import re # regular expression library to handle sentence pattern
    import PyPDF2 # pdf library 
    
    # Create a list to store the extracted text from each page
    sentences = []
    
    ### Part A. Extract text from pdf
    with open(pdf_file, "rb") as f:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        
        # Iteration over each page of the PDF and extract the text content
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
        
            # split the text into sentences
            sentences.extend(text.split('.'))
    
    ### Part B. Clean the sentence
    # Remove any empty sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Remove any /n indicating newline characters
    sentences = [sentence.replace('\n', ' ') for sentence in sentences]
    
    # Replace any mulitple consecutive spaces with a single space '\s+'
    sentences = [re.sub(r'\s+', ' ', sentence) for sentence in sentences]
    
    # Replace any double quotation marks "" with a single
    sentences = [re.sub(r'""', '"', sentence) for sentence in sentences]

    return sentences
    # End


# ## 2. standardize_sentences(extracted_sentences)
# Function: lowering, removing punctuation, numbers, stopwords.

# In[2]:


def standardize_sentences(extracted_sentences):
    """
    Lowercases all texts, removes punctuation, numbers, and stopwords.

    Args:
        extracted_sentences: Extracted and cleaned sentences from the PDF.

    Steps:
        1. Convert text to lowercase
        2. Remove punctuation and numbers
        3. Remove stopwords such as "the," "and," "is," "a," which are commonly used
           without any sentiment.

    Return:
        A list of sentences with all lowercase text,
        without punctuation, numbers, and stopwords.
    """

    import re  # regular expression library to handle sentence pattern
    from nltk.corpus import stopwords  # natural language toolkit stopwords module

    clean_sentences = []  # storing the cleaned sentences
    stop_words = set(stopwords.words('english'))  # using English stopwords

    # iteration over each extracted sentence and then preprocess the sentence
    for sentence in extracted_sentences:
        # Convert text to lowercase
        lower_text = sentence.lower()

        # Remove punctuation and numbers
        # [^a-zA-Z] regular expression for non-English characters
        removed_punc_num = re.sub('[^a-zA-Z]', ' ', lower_text)

        # Remove stopwords
        # iteration over each word and check whether it matches with the stopwords
        # If the word does not match with the stopword, then join into a sentence using a single separator " "
        clean_sentence = ' '.join(word for word in removed_punc_num.split() if word not in stop_words)

        # Append the preprocessed sentence to the list
        clean_sentences.append(clean_sentence)

    return clean_sentences


# ## 3. label_sentences(sentences, word_list)
# Function: labeling the sentences as negative, positive, or neutral based on the given word list.

# In[3]:


def label_sentences(sentences, word_list):
    """
    Labels sentences as negative, positive, or neutral based on the given word list.

    Args:
        sentences: List of preprocessed sentences.
        word_list: DataFrame containing financial words and their sentiment.

    Returns:
        DataFrame with sentences and labels.
    """
    import pandas as pd
    
    labels = []
    positive_words = set(word_list[word_list['sentiment'] == 'positive']['word'])
    negative_words = set(word_list[word_list['sentiment'] == 'negative']['word'])

    for sentence in sentences:
        words = sentence.split()
        if any(word in positive_words for word in words):
            labels.append('positive')
        elif any(word in negative_words for word in words):
            labels.append('negative')
        else:
            labels.append('neutral')

    # Create a DataFrame with sentences and labels
    data = pd.DataFrame({'sentences': sentences, 'labels': labels})

    return data


# ## 4. convert_labels_to_numeric(data, label_mapping)
# 
# Function: Converting categorical labels into numeric values based on the provided label mapping.

# In[4]:


def convert_labels_to_numeric(data, label_mapping):
    """
    Converts categorical labels into numeric values based on the provided label mapping.
    
    Args:
        data (pd.DataFrame): DataFrame containing sentences and labels.
        label_mapping (dict): Mapping of categorical labels to numeric values.
        
    Returns:
        pd.DataFrame: DataFrame with labels converted to numeric values.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    converted_data = data.copy()
    
    # Convert labels to numeric values
    converted_data['labels'] = converted_data['labels'].map(label_mapping)
    
    return converted_data


# ## 5. split_the_dataset(dataset, training_size)
# Function: Spliting the dataset into training and testing sets.

# In[5]:


def split_the_dataset(dataset, training_size):
    """
    Splits the dataset into training and testing sets.

    Args:
        dataset: The dataset to be split.
        training_size: The proportion of the dataset to be used for training.

    Returns:
        Two datasets: the training dataset and the testing dataset.
    """
    # library
    from sklearn.model_selection import train_test_split
    
    # Split the dataset into training and testing sets
    training_data, testing_data = train_test_split(dataset, 
                                                   train_size=training_size, 
                                                   random_state=42)
    
    return training_data, testing_data


# ## 6. tokenize_pad_sequences(sentences, num_words, oov_token, maxlen, padding, truncating)
# Function: Tokenize the text from sentences and pad the sequences

# In[6]:


def tokenize_pad_sequences(sentences, num_words, oov_token, maxlen, padding, truncating):
    """Tokenize the text from sentences and pad the sequences
    Args:
        1. sentences: A list of cleaned and preprocessed sentences.
        2. num_words: An integer specifying the maximum number of words 
        to keep based on word frequency.
        3. oov_token: A string specifying the out-of-vocabulary token to 
        be used for words not present in the tokenizer's word index.
        4. maxlen: An integer specifying the maximum length of the sequences.
        5. padding: A string specifying the padding type to use. It can be either 'pre' or 'post'.
        6. truncating: A string specifying the truncation type to use. It can be 
        either 'pre' or 'post'.
    """
    
    # library
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Initialize the Tokenizer class
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    
    # Split each sentence into words
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Generate indices for each word in the corpus
    tokenizer.fit_on_texts(tokenized_sentences)

    # Get the indices
    word_index = tokenizer.word_index
    
    # Generate list of token sequences
    sequences = tokenizer.texts_to_sequences(tokenized_sentences)
    
    # Pad the sequences with the assigned length, padding, and truncating
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    
    return tokenizer, word_index, sequences, padded_sequences


# ## 7. summary_tokpad(padded_sentences, start_index, end_index)
# Function: Printing a summary of the tokenization and padding process for a given range of entries.

# In[7]:


def summary_tokpad(padded_sentences, start_index, end_index):
    
    """
    Prints a summary of the tokenization and padding process for a given range of entries.

    Args:
        padded_sentences: A tuple containing the word index, sequences, and padded sequences.
        start_index: The starting index of the entries to be summarized.
        end_index: The ending index of the entries to be summarized.

    Returns:
        None (Prints the summary to the console).
    """
    
    # Retrieve the word index, sequences, and padded sequences
    word_index = padded_sentences[1]
    sequences = padded_sentences[2]
    padded_sequences = padded_sentences[3]

    # Print the assigned number of entries of the word index
    print(f"start index: {start_index}")
    print(f"end index: {end_index}")
    print("Summary of the tokenization and padding:")
    print("\nSelected Word Index:")
    for word, index in list(word_index.items())[start_index:end_index]:
        print(f"{word}: {index}")

    # Print the selected sentences
    print("\nSelected Sentences:")
    for sentence in sequences[start_index:end_index]:
        print(sentence)

    # Print the selected padded sequences
    print("\nSelected Padded Sequences:")
    for padded_seq in padded_sequences[start_index:end_index]:
        print(padded_seq)


# ## 8. scrap_yahoo_news(stock_sym, start_date, end_date)
# Function: Scrapes Yahoo Finance news search results for a given query and date range.

# In[8]:


def scrap_yahoo_news(stock_sym, start_date, end_date):
    """
    Scrapes Yahoo Finance news search results for a given query and date range.

    Args:
        stock_sym (str): The stock symbol
        start_date (str): "YYYY-MM-DD"
        end_date (str): "YYYY-MM-DD"

    Returns:
        list: A list of sentences extracted from the news articles.

    """
    # Library
    import requests
    from bs4 import BeautifulSoup
    import re

    # Format the query and date range in the URL
    formatted_query = stock_sym.replace(" ", "+")
    url = f"https://search.yahoo.com/search?p={formatted_query}+news&b={start_date}&bt={end_date}"

    # Send a GET request to the search results page
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the news articles on the search results page
    articles = soup.find_all("div", class_="algo-sr")

    # Create a list to store the extracted content
    contents = []

    # Loop through the articles and extract the content
    for article in articles:
        content = article.find("p").text.strip()
        
        # Remove the date using regex
        content_without_date = re.sub(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}\s*·\s*", "", content)
        
        contents.append(content_without_date)

    # Combine the content into a single string
    combined_content = ' '.join(contents)

    # Split the combined content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', combined_content)

    return sentences


# ## 9. Removing pdf encryption

# In[9]:


def remove_pdf_encryption(input_path, output_path):
    """
    Function to remove security settings from a PDF file.
    
    Args:
        input_path (str): The path to the input PDF file.
        output_path (str): The path to save the output PDF file without security settings.
    """
    import PyPDF2
    
    with open(input_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        if pdf_reader.is_encrypted:
            pdf_reader.decrypt('')
        
        pdf_writer = PyPDF2.PdfWriter()
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)
        
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)

