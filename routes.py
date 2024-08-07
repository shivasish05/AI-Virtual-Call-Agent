
from handlers import twilio_webhook, gather, convo, health_check

def register_routes(app):
    app.add_url_rule('/twilio', 'twilio_webhook', twilio_webhook, methods=['GET', 'POST'])
    app.add_url_rule('/gather', 'gather', gather, methods=['POST'])
    app.add_url_rule('/continue', 'convo', convo, methods=['GET', 'POST'])
    app.add_url_rule('/health_check', 'health_check', health_check)
    # app.add_url_rule('/schedule_meeting', 'schedule_meeting', schedule_meeting, methods=['POST'])

