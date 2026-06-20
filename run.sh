#!/bin/bash

# AuraSense - The big script that sets everything up and runs it
# some pretty colors for the terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}    AuraSense - Employee Activity Monitoring System   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# step 1: make sure mysql is alive and kicking
echo -e "${YELLOW}[1/4] Checking MySQL status...${NC}"
if ! pgrep -x "mysqld" > /dev/null; then
    echo -e "${YELLOW}MySQL not running. Starting via systemctl...${NC}"
    sudo systemctl start mysql
    sleep 3
else
    echo -e "${GREEN}[✓] MySQL server is already running.${NC}"
fi

# step 2: load up our database structure
echo -e "${YELLOW}[2/4] Applying SQL DDL Schema...${NC}"
sudo mysql -e "CREATE DATABASE IF NOT EXISTS employee_activity_db;"
sudo mysql -e "CREATE USER IF NOT EXISTS 'employee_app'@'localhost' IDENTIFIED BY 'aurasense_pass';"
sudo mysql -e "GRANT ALL PRIVILEGES ON employee_activity_db.* TO 'employee_app'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"

if [ $? -eq 0 ]; then
    mysql -u employee_app -paurasense_pass employee_activity_db < schema.sql
    echo -e "${GREEN}[✓] Database schema successfully verified/imported.${NC}"
else
    echo -e "${RED}[✗] Failed to configure MySQL database database connection.${NC}"
fi

# step 3: run a quick health check to make sure python is happy
echo -e "${YELLOW}[3/4] Running diagnostic verification...${NC}"
python3 validate_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}[✗] Diagnostics failed. Please resolve dependencies or database issues.${NC}"
    exit 1
fi

# step 4: fire up the fastapi backend
echo -e "${YELLOW}[4/4] Starting FastAPI Uvicorn Server...${NC}"
echo -e "${GREEN}Dashboard will be available at: http://localhost:8000/${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the monitoring system.${NC}"
echo ""

# pause for a sec then open the browser automatically
(sleep 2 && xdg-open http://localhost:8000/) &

cd backend
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
