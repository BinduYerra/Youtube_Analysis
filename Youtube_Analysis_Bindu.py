#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install google-api-python-client pandas matplotlib


# In[1]:


from googleapiclient.discovery import build

def intilalize_youtube(api_key):
    return build('youtube', 'v3', developerKey=api_key)

youtube = intilalize_youtube('AIzaSyA8orfN9EaN5LzSIuD5peZKkkJzFDVDSvY')


# In[7]:


get_ipython().system('pip install isodate')


# In[2]:


import isodate

def get_trending_videos(youtube, region_code='US', category_id=None, max_videos=200):
    all_videos = []
    next_page_token = None
    videos_fetched = 0

    while videos_fetched < max_videos:
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart='mostPopular',
            regionCode=region_code,
            videoCategoryId=category_id,  
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        # Process and filter videos where duration is more than 120 seconds to exclude shorts
        for item in video_response.get('items', []):
            duration = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
            if duration > 120:  # Exclude shorts by duration
                all_videos.append(item)
        
        videos_fetched = len(all_videos)
        next_page_token = video_response.get('nextPageToken')
        if not next_page_token:
            break

    return {'items': all_videos[:max_videos]}  # Limit to max_videos


# In[3]:


from googleapiclient.errors import HttpError


def fetch_and_process_all_videos(youtube):
    categories = get_video_categories(youtube)
    all_videos_df = pd.DataFrame()

    for category_id, category_name in categories.items():
        try:
            # Fetch trending videos for this category (limit to 200)
            video_response = get_trending_videos(youtube, category_id=category_id, max_videos=200)
            videos_df = process_videos_data(video_response, category_name)
            all_videos_df = pd.concat([all_videos_df, videos_df], ignore_index=True)

            # Stop if total videos exceed 200
            if len(all_videos_df) >= 200:
                all_videos_df = all_videos_df.head(200)  # Ensure only the top 200 videos
                break
        except HttpError as e:
            print(f"Skipping category {category_name} (ID: {category_id}) due to error: {e}")

    return all_videos_df


# In[4]:


# Fetch available categories in the US
def get_video_categories(youtube, region_code='US'):
    category_response = youtube.videoCategories().list(
        part="snippet",
        regionCode=region_code
    ).execute()
    
    # Map category IDs to category names
    categories = {item['id']: item['snippet']['title'] for item in category_response['items']}
    return categories

# Fetch categories and store them globally
categories_mapping = get_video_categories(youtube)
print("Available Categories:\n", categories_mapping)


# In[5]:


import pandas as pd

def process_videos_data(video_response, category_name):
    videos = []
    for item in video_response.get('items', []):
        video_id = item['id']
        title = item['snippet']['title']
        description = item['snippet']['description']
        published_at = item['snippet']['publishedAt']
        channel_id = item['snippet']['channelId']
        channel_title = item['snippet']['channelTitle']
        category_id = item['snippet'].get('categoryId', 'N/A') 
        category_name = categories_mapping.get(category_id, 'Unknown')  # Map category name
        tags = item['snippet'].get('tags', [])
        duration = item['contentDetails'].get('duration', 'N/A')  
        definition = item['contentDetails'].get('definition', 'N/A')
        caption = item['contentDetails'].get('caption', 'false')
        views = item['statistics'].get('viewCount', 0)
        likes = item['statistics'].get('likeCount', 0)
        dislikes = item['statistics'].get('dislikeCount', 0)
        comments = item['statistics'].get('commentCount', 0)

        videos.append({
            'Video ID': video_id,
            'Title': title,
            'Description': description,
            'Published At': published_at,
            'Channel ID': channel_id,
            'Category Id': category_id,
            'Channel Title': channel_title,
            'Category': category_name,
            'Tags': tags,
            'Duration': duration,
            'Definition': definition,
            'Caption': caption,
            'Views': int(views),
            'Likes': int(likes),
            'Dislikes': int(dislikes),
            'Comments': int(comments)
        })

    return pd.DataFrame(videos)
        


# In[7]:


all_videos_df = fetch_and_process_all_videos(youtube)
all_videos_df.to_csv('trending_videos.csv', index=False)
print("Data saved to 'trending_videos.csv'.")


# In[8]:


trending_videos = pd.read_csv('trending_videos.csv')
print(trending_videos.head())


# In[9]:


# check for missing values
missing_values = trending_videos.isnull().sum()

# display data types
data_types = trending_videos.dtypes

missing_values, data_types


# In[10]:


# fill missing descriptions with "No description"
trending_videos['Description'].fillna('No description', inplace=True)

# convert `published_at` to datetime
trending_videos['Published At'] = pd.to_datetime(trending_videos['Published At'])

# convert tags from string representation of list to actual list
trending_videos['Tags'] = trending_videos['Tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)


# In[11]:


# check for missing values
missing_values = trending_videos.isnull().sum()

# display data types
data_types = trending_videos.dtypes

missing_values, data_types


# In[12]:


# descriptive statistics
descriptive_stats = trending_videos[['Views', 'Likes', 'Dislikes', 'Comments']].describe()

descriptive_stats


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# view count distribution
sns.histplot(trending_videos['Views'], bins=30, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Views Distribution')
axes[0].set_xlabel('Views')
axes[0].set_ylabel('Frequency')

# like count distribution
sns.histplot(trending_videos['Likes'], bins=30, kde=True, ax=axes[1], color='green')
axes[1].set_title('Likes Distribution')
axes[1].set_xlabel('Likes')
axes[1].set_ylabel('Frequency')

# comment count distribution
sns.histplot(trending_videos['Comments'], bins=30, kde=True, ax=axes[2], color='red')
axes[2].set_title('Comments Distribution')
axes[2].set_xlabel('Comments')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[14]:


# correlation matrix
correlation_matrix = trending_videos[['Views', 'Likes', 'Comments']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix of Engagement Metrics')
plt.show()


# In[15]:


# Calculate engagement ratios
trending_videos['Like-to-View Ratio'] = trending_videos['Likes'] / trending_videos['Views']
trending_videos['Dislike-to-View Ratio'] = trending_videos['Dislikes'] / trending_videos['Views']
trending_videos['Comment-to-View Ratio'] = trending_videos['Comments'] / trending_videos['Views']

# Summarize the impact of video features like HD and captions
caption_impact = trending_videos.groupby('Caption').mean(numeric_only=True)[['Views', 'Like-to-View Ratio']]
hd_impact = trending_videos.groupby('Definition').mean(numeric_only=True)[['Views', 'Like-to-View Ratio']]
print("Impact of Captions:\n", caption_impact)
print("Impact of HD Quality:\n", hd_impact)


# In[16]:


# Time-based analysis
trending_videos['Day of Week'] = trending_videos['Published At'].dt.day_name()
views_by_day = trending_videos.groupby('Day of Week').mean(numeric_only=True)['Views']
print("Average Views by Day of Week:\n", views_by_day)


# In[17]:


import seaborn as sns

# Group by Day of Week and Category to calculate average views
category_day_analysis = trending_videos.groupby(['Day of Week', 'Category']).mean(numeric_only=True)['Views'].unstack()

# Reset index for better visualization
category_day_analysis = category_day_analysis.reset_index()

# Plot a heatmap to visualize trends
plt.figure(figsize=(20, 16))
sns.heatmap(
    category_day_analysis.set_index('Day of Week'),
    annot=True,
    fmt=".0f",
    cmap='Blues'
)
plt.title('Average Views by Content Type and Day of Week')
plt.ylabel('Day of Week')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.show()


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

# Vectorize the titles to find common words
vectorizer = CountVectorizer(stop_words='english', max_features=100)
titles_vectorized = vectorizer.fit_transform(trending_videos['Title'])
words_frame = pd.DataFrame(titles_vectorized.toarray(), columns=vectorizer.get_feature_names_out())

# Analyze tags
trending_videos['Tag Count'] = trending_videos['Tags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
tag_view_correlation = trending_videos[['Tag Count', 'Views']].corr()
print("Correlation between Number of Tags and Views:\n", tag_view_correlation)

# Adding word counts to the main DataFrame to analyze correlation
trending_videos = pd.concat([trending_videos, words_frame], axis=1)
word_view_correlation = trending_videos.corr(numeric_only=True)['Views'].sort_values(ascending=False)
print("Correlation of Words in Titles with Views:\n", word_view_correlation.head(20))


# In[19]:


# Plot the average views by day of the week
plt.figure(figsize=(10, 6))
plt.plot(views_by_day.index, views_by_day.values, marker='o', linestyle='-')
plt.title('Average Views by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Views')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[20]:


plt.figure(figsize=(12, 8))
sns.countplot(y='Category', data=all_videos_df, order=all_videos_df['Category'].value_counts().index, palette='viridis')
plt.title('Number of Trending Videos by Category')
plt.xlabel('Number of Videos')
plt.ylabel('Category')
plt.show()


# In[21]:


# average engagement metrics by category
category_engagement = trending_videos.groupby('Category')[['Views', 'Likes', 'Comments']].mean().sort_values(by='Views', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# view count by category
sns.barplot(y=category_engagement.index, x=category_engagement['Views'], ax=axes[0], palette='viridis')
axes[0].set_title('Average Views by Category')
axes[0].set_xlabel('Average Views')
axes[0].set_ylabel('Category')

# like count by category
sns.barplot(y=category_engagement.index, x=category_engagement['Likes'], ax=axes[1], palette='viridis')
axes[1].set_title('Average Likes Category')
axes[1].set_xlabel('Average Likes')
axes[1].set_ylabel('')

# comment count by category
sns.barplot(y=category_engagement.index, x=category_engagement['Comments'], ax=axes[2], palette='viridis')
axes[2].set_title('Average Comments Category')
axes[2].set_xlabel('Average Comments')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()


# In[22]:


# Convert ISO 8601 duration to seconds and handle exceptions for any improperly formatted entries
trending_videos['duration_seconds'] = trending_videos['Duration'].apply(lambda x: isodate.parse_duration(x).total_seconds() if pd.notna(x) else 0)

# Categorize the duration into bins
trending_videos['duration_range'] = pd.cut(
    trending_videos['duration_seconds'],
    bins=[0, 300, 600, 1200, 3600, 7200],
    labels=['0-5 min', '5-10 min', '10-20 min', '20-60 min', '60-120 min']
)


# In[23]:


print(trending_videos['duration_seconds'])


# In[24]:


# Scatter plot for video length vs view count
plt.figure(figsize=(12, 6))
sns.scatterplot(x='duration_seconds', y='Views', data=trending_videos, alpha=0.6, color='purple')
plt.title('Video Length vs Views')
plt.xlabel('Video Length (seconds)')
plt.ylabel('Views')
plt.show()


# In[26]:


# bar chart for engagement metrics by duration range
length_engagement = trending_videos.groupby('duration_range')[['Views', 'Likes', 'Comments']].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# view count by duration range
sns.barplot(y=length_engagement.index, x=length_engagement['Views'], ax=axes[0], palette='magma')
axes[0].set_title('Average Views by Duration Range')
axes[0].set_xlabel('Average Views')
axes[0].set_ylabel('Duration Range')

# like count by duration range
sns.barplot(y=length_engagement.index, x=length_engagement['Likes'], ax=axes[1], palette='magma')
axes[1].set_title('Average Likes by Duration Range')
axes[1].set_xlabel('Average Likes')
axes[1].set_ylabel('')

# comment count by duration range
sns.barplot(y=length_engagement.index, x=length_engagement['Comments'], ax=axes[2], palette='magma')
axes[2].set_title('Average Comments by Duration Range')
axes[2].set_xlabel('Average Comments')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()


# In[29]:


# calculate the number of tags for each video
trending_videos['Tags'] = trending_videos['Tags'].apply(len)

# scatter plot for number of tags vs view count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Tags', y='Views', data=trending_videos, alpha=0.6, color='orange')
plt.title('Number of Tags vs Views')
plt.xlabel('Number of Tags')
plt.ylabel('Views')
plt.show()


# In[30]:


# extract hour of publication
trending_videos['publish_hour'] = trending_videos['Published At'].dt.hour

# bar chart for publish hour distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='publish_hour', data=trending_videos, palette='coolwarm')
plt.title('Distribution of Videos by Publish Hour')
plt.xlabel('Publish Hour')
plt.ylabel('Number of Videos')
plt.show()

# scatter plot for publish hour vs view count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='publish_hour', y='Views', data=trending_videos, alpha=0.6, color='teal')
plt.title('Publish Hour vs Views')
plt.xlabel('Publish Hour')
plt.ylabel('Views')
plt.show()


# In[31]:


get_ipython().system('pip install textblob')
get_ipython().system('python -m textblob.download_corpora')


# In[32]:


from textblob import TextBlob

def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Return polarity and subjectivity
    return blob.sentiment.polarity, blob.sentiment.subjectivity


# In[33]:


# Apply sentiment analysis to titles
trending_videos['Title Sentiment'] = trending_videos['Title'].apply(lambda x: analyze_sentiment(x)[0])
trending_videos['Title Subjectivity'] = trending_videos['Title'].apply(lambda x: analyze_sentiment(x)[1])

# Apply sentiment analysis to descriptions
trending_videos['Description Sentiment'] = trending_videos['Description'].apply(lambda x: analyze_sentiment(x)[0])
trending_videos['Description Subjectivity'] = trending_videos['Description'].apply(lambda x: analyze_sentiment(x)[1])


# In[34]:


# Example: Analyze average sentiment by views
average_sentiment_by_views = trending_videos.groupby(pd.cut(trending_videos['Views'], bins=5)).agg({
    'Title Sentiment': 'mean',
    'Description Sentiment': 'mean'
}).reset_index()

print(average_sentiment_by_views)


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(x='Views', y='Title Sentiment', data=average_sentiment_by_views)
plt.title('Average Sentiment of Video Titles by Views')
plt.xlabel('Views (binned)')
plt.ylabel('Average Sentiment')
plt.show()


# In[ ]:




