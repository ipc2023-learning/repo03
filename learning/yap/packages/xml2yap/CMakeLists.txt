  

INCLUDE_DIRECTORIES(
  ../../H
  ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/CXX
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/H
    ${CMAKE_SOURCE_DIR}/OPTYap
    )

   )



  SET ( RAPTOR2_SOURCES
    pugixml.cpp
xml2pl.cpp
    )


  if(DEFINED YAP_MAJOR_VERSION)
    TARGET_LINK_LIBRARIES(PUGIXML
      ${PUGIXML_LIBRARY}
      libYap
      )
  else()
    TARGET_LINK_LIBRARIES(PUGIXML
      ${PUGIXML_LIBRARY}
      ${YAP_LIBRARY}
      )
  endif()


  install(TARGETS  PUGIXML
          LIBRARY DESTINATION ${YAP_INSTALL_DLLDIR}
          RUNTIME DESTINATION ${YAP_INSTALL_DLLDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}

    )

INSTALL(FILES xml2yap.yap DESTINATION ${YAP_INSTALL_DATADIR})

