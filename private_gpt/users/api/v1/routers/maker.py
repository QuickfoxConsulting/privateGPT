from typing import Any, List
import datetime
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request
from pathlib import Path
from private_gpt.server.ingest.ingest_router import ingest_file
import os
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas
import traceback
from private_gpt.users.constants.makerchecker import MakerCheckerStatus, MakerCheckerActionType
from private_gpt.users.constants.upload_type import UploadType
from private_gpt.constants import UPLOAD_DIR, OCR_UPLOAD, MAKER_UPLOAD

router = APIRouter(prefix="/maker", tags=["Maker"])


@router.post('/verify')
async def verify(
    request: Request,
    verify_in: schemas.CheckerUpdate,
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"], Role.OPERATOR["name"]],
    )
):
    try:
        if not (current_user.checker and (checker.created_by != current_user.id)):
            raise HTTPException(status_code=403, detail="Forbidden")
        
        checker = crud.makerchecker.get_by_id(db, id=verify_in.id)
        if not checker:
            raise HTTPException(status_code=404, detail="Oject not found")
        
        if verify_in.table_name == 'document':
            doc = crud.documents.get_by_id(db, id=verify_in.record_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            upload_type = doc.upload_type
            file_name = doc.filename
            unchecked_filepath = Path(f"{MAKER_UPLOAD}/{file_name}")

            if verify_in.status == MakerCheckerStatus.APPROVED:
                checker_in = schemas.MakerCheckerUpdate(
                    action_type=MakerCheckerActionType.UPDATE,
                    status=MakerCheckerStatus.APPROVED,
                    verified_at=datetime.datetime.now(),
                    verified_by=current_user.id
                )
                crud.makerchecker.update(db, db_obj=checker, obj_in=checker_in)
                
                if upload_type == UploadType.REGULAR:
                    response = ingest_file(request=request, db=db, original_file=unchecked_filepath,log_audit=log_audit)
                    os.remove(unchecked_filepath)
                else:
                    pass     

            elif verify_in.status == MakerCheckerStatus.REJECTED:
                checker_in = schemas.MakerCheckerUpdate(
                    action_type=MakerCheckerActionType.UPDATE,
                    status=MakerCheckerStatus.APPROVED,
                    verified_at=datetime.datetime.now(),
                    verified_by=current_user.id
                )
                crud.makerchecker.update(db, db_obj=checker, obj_in=checker_in)
                crud.documents.remove(db=db, id=doc.id)
                os.remove(unchecked_filepath)

            else:
                pass
        else:
            pass
        return "Verified"
    
    except HTTPException:
        print(traceback.print_exc())
        raise

    except Exception as e:
        print(traceback.print_exc())
        log_audit(
            model="Document",
            action="create",
            details={
                "status": 500,
                "detail": "Internal Server Error: Unable to ingest file.",
            },
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Unable to ingest file.",
        )


    
