# Writes OUTPUT, a header with the git state of SOURCE_DIR, at every build;
# the file changes (and dependents rebuild) only when the state does.
set(describe "")
set(sha "")
set(dirty false)
if(GIT_EXECUTABLE)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${SOURCE_DIR} OUTPUT_VARIABLE describe OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${SOURCE_DIR} OUTPUT_VARIABLE sha OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(describe MATCHES "-dirty$")
        set(dirty true)
    endif()
endif()
file(WRITE ${OUTPUT}.tmp
"#ifndef MALLARD_GIT_VERSION_H
#define MALLARD_GIT_VERSION_H

#define MALLARD_GIT_DESCRIBE \"${describe}\"
#define MALLARD_GIT_SHA \"${sha}\"
#define MALLARD_GIT_DIRTY ${dirty}

#endif // MALLARD_GIT_VERSION_H
")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OUTPUT}.tmp ${OUTPUT})
file(REMOVE ${OUTPUT}.tmp)
