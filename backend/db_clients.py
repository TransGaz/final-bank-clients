
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase): pass


class Client(Base):
    __tablename__ = "clients"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(primary_key=True, name="id")
    agreement_rk: Mapped[int] = mapped_column(nullable=False, name='agreement_rk')
    age: Mapped[int] = mapped_column(nullable=False, name='age')
    socstatus_work_fl: Mapped[int] = mapped_column(nullable=False, name='socstatus_work_fl')
    socstatus_pens_fl: Mapped[int] = mapped_column(nullable=False, name='socstatus_pens_fl')
    gender: Mapped[int] = mapped_column(nullable=False, name='gender')
    child_total: Mapped[int] = mapped_column(nullable=False, name='child_total')
    dependants: Mapped[int] = mapped_column(nullable=False, name='dependants')
    personal_income: Mapped[float] = mapped_column(nullable=False, name='personal_income')
    loan_num_total: Mapped[int] = mapped_column(nullable=False, name='loan_num_total')
    loan_num_closed: Mapped[int] = mapped_column(nullable=False, name='loan_num_closed')
    target: Mapped[int] = mapped_column(nullable=False, name='target')



