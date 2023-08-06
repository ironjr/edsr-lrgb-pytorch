#!/bin/bash
curl https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Messages.json -X POST \
    --data-urlencode 'To='$MY_PHONE_NUMBER --data-urlencode 'From='$TWILIO_PHONE_NUMBER \
    --data-urlencode 'Body='"$1" -u $TWILIO_ACCOUNT_SID:$TWILIO_AUTHTOKEN
