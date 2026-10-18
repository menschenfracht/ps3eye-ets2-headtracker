PS3 Eye DirectShow 120 FPS - Portable x64 Package
=================================================

This package provides a 64-bit DirectShow source filter for the PlayStation 3
Eye camera. It is intended for 64-bit applications on 64-bit Windows only.

Project source and releases:
https://github.com/visapex-app/PS3EyeDirectShow-120fps

VisApex website:
https://visapex.app


Open-source software
--------------------

This package is open-source software licensed under GPL-3.0-only. The complete
project source is available at the project URL above. See LICENSE.txt for the
license terms. Forks and modified distributions can continue using the same
tag-based build and packaging workflow; distributors must provide the
corresponding source and retain the applicable notices as required by the GPL.


Before installing
-----------------

This package contains the DirectShow filter, not the USB kernel driver. The
user-mode libusb library is included in PS3EyeSourceFilter.dll, but Windows
must still have a compatible driver bound to the physical camera.

If the camera already works with the Code Laboratories CL-Eye driver, keep the
existing driver. It is compatible with this filter and does not need to be
replaced or reinstalled. Register only the x64 DirectShow filter supplied in
this package to make the camera available to compatible 64-bit applications.

Code Laboratories CL-Eye website:
https://codelaboratories.com/

On a PC without a working PS3 Eye driver, install a compatible USB driver
before registering the filter. Valid options include:

- Code Laboratories CL-Eye driver:
  https://codelaboratories.com/
- libusbK using the PS3 Eye open-driver instructions from opentrack:
  https://github.com/opentrack/opentrack/wiki/PS3-Eye-open-driver-instructions

Close any application that is using the PS3 Eye before registering,
unregistering, updating, or replacing the filter.


Install / register
------------------

1. Extract every file in this ZIP to a permanent folder of your choice.
2. Keep PS3EyeSourceFilter.dll and both batch files together in that folder.
3. Right-click Register-PS3EyeDirectShow.bat and select "Run as administrator".
4. Accept the Windows User Account Control prompt if one appears.
5. A successful RegSvr32 message confirms that the filter is registered.

Do not move, rename, or delete the extracted folder after registration.
Windows records the DLL's current full path. Moving or deleting it will break
the registered filter.


Uninstall / unregister
----------------------

1. Close all applications that might be using the camera.
2. Right-click Unregister-PS3EyeDirectShow.bat and select
   "Run as administrator".
3. Wait for the successful RegSvr32 message.
4. You may then delete the extracted folder.


Update
------

1. Run Unregister-PS3EyeDirectShow.bat as administrator from the old folder.
2. Extract the new package to a permanent folder.
3. Run Register-PS3EyeDirectShow.bat as administrator from the new folder.


Notes
-----

- This x64 filter is visible only to 64-bit DirectShow applications.
- Registration modifies the Windows machine-wide DirectShow/COM registry and
  therefore requires administrator permission.
- If registration reports that access was denied, verify that the batch file
  was run as administrator.

See LICENSE.txt for the software license.
