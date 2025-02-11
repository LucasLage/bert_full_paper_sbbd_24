# -*- coding: utf-8 -*-
import logging
import utils.preprocessing_portuguese as preprossPT
import utils.preprocessing_module as preprocess
import pandas as pd
import nltk
import csv
import re
import psycopg2

from tqdm import tqdm


import spacy

def limpeza_texto(page_text):
    txt_process = preprossPT.TextPreProcessing()

    #     page_text = txt_process.remove_person_names(page_text) 
    page_text = page_text.lower()    
    page_text = txt_process.remove_emails(page_text)
    page_text = txt_process.remove_urls(page_text)    
    #     page_text = txt_process.remove_pronouns(page_text)
    #     page_text = txt_process.remove_adverbs(page_text)
    page_text = txt_process.remove_special_characters(page_text)
    page_text = txt_process.remove_accents(page_text)
    page_text = txt_process.remove_stopwords(page_text)
    page_text = txt_process.remove_hour(page_text)

    # split numbers from letters
    page_text = ' '.join(re.split('(\d+)',page_text))
    page_text = txt_process.remove_symbols_from_numbers(page_text)
    page_text = txt_process.remove_numbers(page_text)
    #     page_text = txt_process.remove_reduced_or_contracted_words(page_text)

    #Removendo letras sozinhas no texto
    #     page_text = re.sub(r'(?:^| )\w(?:$| )', ' ', page_text).strip()
    page_text = re.sub(r"\b[a-zA-Z]\b", "", page_text)
    page_text = page_text.replace("_","")

    #     words_to_remove = ["tel", "fax", "cnpj", "cpf", "mail", "cep", "estado", "minas gerais", "prefeitura", "municipal", "municipio"]
    #     for word in words_to_remove:
    #         page_text = page_text.replace(word,"")

    page_text = txt_process.remove_excessive_spaces(page_text)    
    return page_text

def limpeza_texto_unit_entity(page_text):
    txt_process = preprossPT.TextPreProcessing()    
    page_text = page_text.lower()    
    page_text = txt_process.remove_entities(page_text)    
    page_text = txt_process.remove_units(page_text)

    #Remover palavras com digitos
    page_text = ' '.join(w for w in page_text.split() if not any(x.isdigit() for x in w))    
    page_text = txt_process.remove_person_names(page_text)    
    page_text = txt_process.remove_emails(page_text)
    page_text = txt_process.remove_urls(page_text)    
    page_text = txt_process.remove_pronouns(page_text)
    page_text = txt_process.remove_adverbs(page_text)
    page_text = txt_process.remove_special_characters(page_text)
    page_text = txt_process.remove_accents(page_text)
    page_text = txt_process.remove_stopwords(page_text)
    page_text = txt_process.remove_hour(page_text)

    # split numbers from letters
    #page_text = ' '.join(re.split('(\d+)',page_text))
    page_text = txt_process.remove_symbols_from_numbers(page_text)
    page_text = txt_process.remove_numbers(page_text)
    page_text = txt_process.remove_reduced_or_contracted_words(page_text)

    #Removendo letras sozinhas no texto
    page_text = re.sub(r"\b[a-zA-Z]\b", "", page_text)    
    page_text = page_text.replace("_","")    

    page_text = txt_process.remove_excessive_spaces(page_text)    
    return page_text

def preprocess_text_complete(document, process_type, key):
    if process_type == 'traditional':
        return  limpeza_texto(document['conteudo'])
    elif process_type == 'simple':
        return [preprocess.limpeza_texto(page_content, [], "simple") for page_content in document[key]]
    elif process_type == 'simple-unity':
        return [preprocess.limpeza_texto(page_content, [], "simple-unity") for page_content in document[key]]
    elif process_type == 'complete':
        return [preprocess.limpeza_texto(page_content, [], "complete") for page_content in document[key]]
    elif process_type == 'complete-unity':
        return [preprocess.limpeza_texto(page_content, [], "complete-unity") for page_content in document[key]]        
    else: # apenas unidades de medida e entidades
        return [limpeza_texto_unit_entity(page_content) for page_content in document[key]]


def process_data(df_path):
    
    df = pd.read_csv(df_path)
    print(len(df))
    print(df['text_masked'].isna().sum())
    df = df.dropna()
    print(len(df))
    
    count = len(df)
    
    process_type = 'traditional'
    columns = [
        'uuid_segmento',
        'file_path_datalake',
        'id_segmento',
        'orgao',
        'classe_modalidade',
        'text_complete',
        'title',
        'subtitle',
        'content',
        'text_masked',
        'preprocessed_text',
        'preprocessed_title',
        'preprocessed_subtitle',
        'preprocessed_content',
        'preprocessed_masked',
        'num_sentences_text',
        'num_sentences_title',
        'num_sentences_subtitle',
        'num_sentences_content',
        'num_sentences_masked',
        'avg_len_sentence_text',
        'avg_len_sentence_title',
        'avg_len_sentence_subtitle',
        'avg_len_sentence_content',
        'avg_len_sentence_masked',
        'text_size_text',
        'text_size_title',
        'text_size_subtitle',
        'text_size_content',
        'text_size_masked',
        'preprocessed_text_size',
        'preprocessed_text_title_size',
        'preprocessed_text_subtitle_size',
        'preprocessed_text_content_size',
        'preprocessed_text_masked_size'
    ]
    with open('./preprocessed_data.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for index, row in tqdm(df.iterrows(), unit='segments preprocessed', total=count):
            text_complete = row['full_text']
            title = row['titulo']
            subtitle = row['subtitulo']
            content = row['conteudo']
            # classe_modalidade = row["classe_modalidade"]
            text_masked = row['text_masked']
            tokens_text = nltk.sent_tokenize(text_complete)
            tokens_title = nltk.sent_tokenize(title)
            tokens_subtitle = nltk.sent_tokenize(subtitle)
            tokens_content = nltk.sent_tokenize(content)
            tokens_masked = nltk.sent_tokenize(text_masked)
            num_sentences_text = len(tokens_text)
            num_sentences_title = len(tokens_title)
            num_sentences_subtitle = len(tokens_subtitle)
            num_sentences_content = len(tokens_content)
            num_sentences_masked = len(tokens_masked)
            try:
                avg_len_sentence_text = sum( map(len, tokens_text) ) / num_sentences_text
                avg_len_sentence_title = sum( map(len, tokens_title) ) / num_sentences_title
                avg_len_sentence_subtitle = sum( map(len, tokens_subtitle) ) / num_sentences_subtitle
                avg_len_sentence_content = sum( map(len, tokens_content) ) / num_sentences_content
                avg_len_sentence_masked = sum( map(len, tokens_masked) ) / num_sentences_masked
                
            except:
                avg_len_sentence_text = len(tokens_text)
                avg_len_sentence_title = len(tokens_title)
                avg_len_sentence_subtitle = len(tokens_subtitle)
                avg_len_sentence_content = len(tokens_content)
                avg_len_sentence_masked = len(tokens_masked)
                
            preprocessed_text = preprocess_text_complete(row, process_type, key='full_text') 
            preprocessed_title = preprocess_text_complete(row, process_type, key='titulo') 
            preprocessed_subtitle = preprocess_text_complete(row, process_type, key='subtitulo') 
            preprocessed_content = preprocess_text_complete(row, process_type, key='conteudo') 
            preprocessed_masked = preprocess_text_complete(row, process_type, key='text_masked') 
            
            

            new_row = [
                row['uuid_segmento'], 
                row['file_path_datalake'],
                row['id_segmento'],
                row['orgao'],
                row['classe_modalidade'],
                text_complete,
                title,
                subtitle,
                content,
                text_masked,
                preprocessed_text, 
                preprocessed_title,
                preprocessed_subtitle, 
                preprocessed_content, 
                preprocessed_masked,
                num_sentences_text, 
                num_sentences_title,
                num_sentences_subtitle,
                num_sentences_content,
                num_sentences_masked,
                avg_len_sentence_text,
                avg_len_sentence_title,
                avg_len_sentence_subtitle,
                avg_len_sentence_content,
                avg_len_sentence_masked,
                len(text_complete),
                len(title),
                len(subtitle),
                len(content),
                len(text_masked),
                len(preprocessed_text),
                len(preprocessed_title),
                len(preprocessed_subtitle),
                len(preprocessed_content),
                len(preprocessed_masked)]


            writer.writerow(new_row)

    file_name = "preprocessed_data.csv"
    print(f"Preprocess Finished. preprocessed data saved on {file_name}")
    logging.info(f"Preprocess Finished. preprocessed data saved on {file_name}")
    return file_name