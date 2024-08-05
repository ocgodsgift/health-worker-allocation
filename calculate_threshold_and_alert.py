import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

def trigger(data, keywords):
    # Alert threshold setup
    threshold = 6
    time_frame = timedelta(weeks=1)
    current_date = datetime.now()

    # Email credentials
    email_host = 'smtp.gmail.com'
    email_port = 587
    email_username = 'your_email@example.com'
    email_password = 'your_password'
    recipients = 'recipient@example.com'   

    # Function to send email
    def send_email(subject, body):
        msg = MIMEMultipart()
        msg['From'] = email_username
        msg['To'] = recipients
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(email_host, email_port)
        server.starttls()
        server.login(email_username, email_password)
        text = msg.as_string()
        server.sendmail(email_username, recipients, text)
        server.quit()

    # Check for recent occurrences exceeding the threshold and display alerts
    for keyword in keywords:
        recent_count = data[(data['date'] >= current_date - time_frame) & (data['comment'].str.contains(keyword, case=False, na=False))].shape[0]
        if recent_count >= threshold:
            # Email body
            subject = "Early Warning System ALERT!!"
            body = f"""
            ALERT: {keyword.capitalize()} has been reported {recent_count} times in the last week!

            Please reach out to the EdoDiDa Team for more information.

            Best Regards,
            EdoDiDa Team.
            """
            
            send_email(subject, body)
