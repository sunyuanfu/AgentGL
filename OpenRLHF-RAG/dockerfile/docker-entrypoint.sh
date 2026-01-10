#!/bin/bash

USER=${LOCAL_USER:-"root"}

if [[ "${USER}" != "root" ]]; then
    USER_ID=${LOCAL_USER_ID:-9001}
    echo ${USER}
    echo ${USER_ID}

    chown ${USER_ID} /PATH/TO/HOME/${USER}
    useradd --shell /bin/bash -u ${USER_ID} -o -c "" -m ${USER}
    usermod -a -G root ${USER}
    adduser ${USER} sudo

    # user:password
    echo "${USER}:123" | chpasswd

    export HOME=/PATH/TO/HOME/${USER}
    export PATH=/PATH/TO/HOME/${USER}/.local/bin/:$PATH
else
    export PATH=/PATH/TO/ROOT/.local/bin/:$PATH
fi

cd $HOME
exec gosu ${USER} "$@"