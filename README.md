# text_message_analysis
Analyze your IOs messages from your device with Pandas/NLTK.

I wrote this over a couple caffeinated hours. There are no classes -- it's just
straight procedural coding. There are a few issues, but I haven't fixed them. 
One issue, for example is the timestamp conversion method I currently use
yields dates into 2016... it's currently 2015. Let's go back to the future!!

Additionally, if you read through the code, you can see that when I tokenize 
all of the messages, I actually tokenize them when they are in a massive 
message string. I chose to concatenate every single text message into a 
"corpus." You don't have to do it this way. Instead, you can iterate through
all of the messages and tokenzie each one individually. 

If you like this code, or want to fix things, feel free.

And you can read more about this project on my blog here:
http://sweet-as-tandy.com/2015/06/26/how-to-retrieve-and-analyze-your-ios-messages-with-python-pandas-and-nltk/
