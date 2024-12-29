# Initialisation 

## Creating a mysql image
sudo docker run --name mysql_name -e MYSQL_ROOT_PASSWORD=create_password -d mysql:latest

## Creating adminer for gui 
sudo docker run -d --name admin_name -p 8080:8080 --link mysql_name:mysql adminer

# For checking docker image
sudo docker ps 
sudo docker ps -a

# For manipulation in created mysql_name
sudo docker exec -it mysql_name mysql -uroot -p

sudo docker restart admin_name
(If admin is not changed)

# Log in Adminer
- Server: mysql (--link is aliased as mysql)
- Username: root (or any other username)
- Password: create_password

`Port: localhost:8080`

# For stoping 
sudo docker stop mysql_name

sudo docker stop admin_name

# For uploading a predefined database 

create a copy of sql dump file `*.sql`
Run:
sudo docker exec -i mysql_name mysql -uroot -pmy-secret-pw < /path/to/datbase.sql
