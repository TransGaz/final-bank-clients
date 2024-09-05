from sqlalchemy import URL

#SQLALCHEMY_DATABASE_URL = "sqlite:///./clients4g.db"
#SQLALCHEMY_DATABASE_URL = "postgresql://bankmarketingdb_owner:wfvs4DqVOBp9@ep-plain-tree-a2tvinw3.eu-central-1.aws.neon.tech/bankmarketingdb?sslmode=require"

connection_string = URL.create(
  'postgresql',
  username='bankmarketingdb_owner',
  password='wfvs4DqVOBp9',
  host='ep-plain-tree-a2tvinw3.eu-central-1.aws.neon.tech',
  database='bankmarketingdb',
  #connect_args={'sslmode':'require'}
)