_f90_verbose = $(_f90_verbose_$(V))
_f90_verbose_ = $(_f90_verbose_$(AM_DEFAULT_VERBOSITY))
_f90_verbose_0 = @echo "  $1";

# $1 program
define is_per_target
$(if $(filter $(patsubst %.F90,%.o,$(firstword $($1_SOURCES))),$($1_OBJECTS)),,1)
endef

# $1 source_file
# $2 stem
# $3 program
define module_targets
$(eval _$(3)_use_mods += $(dir $1)$(2)$(patsubst %.F90,%,$(notdir $1)).use_mods)
$(dir $1)$(2)$(patsubst %.F90,%,$(notdir $1)).use_mods: $1 $(dir $1)$(am__dirstamp)
	$(call _f90_verbose,F90 USE  [$3] $$<)$(CPP) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $($p_CPPFLAGS) $(CPPFLAGS) -o /dev/stdout $$< | grep -i -o '^ *use [^ ,!:]*' | sort -u > $$@

$(eval _$(3)_def_mods += $(dir $1)$(2)$(patsubst %.F90,%,$(notdir $1)).def_mods)
$(dir $1)$(2)$(patsubst %.F90,%,$(notdir $1)).def_mods: $1 $(dir $1)$(am__dirstamp)
	$(call _f90_verbose,F90 MOD  [$3] $$<)$(CPP) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $($p_CPPFLAGS) $(CPPFLAGS) -o /dev/stdout $$< | grep -i -o '^ *module [^!]*' > $$@ || true
endef
$(foreach p,$(bin_PROGRAMS),$(if $(call is_per_target,$p),$(foreach s,$($p_SOURCES),$(eval $(call module_targets,$s,$p-,$p))),$(foreach s,$($p_SOURCES),$(eval $(call module_targets,$s,,$p)))))

_f90_moddir=$(abs_builddir)/.fortran_modules
_f90_depfile = $(_f90_moddir)/dependencies.mk

define is_clean
$(if $(filter-out mostlyclean clean distclean maintainer-clean,$(MAKECMDGOALS)),0,1)
endef

ifneq ($(call is_clean),1)
include $(_f90_depfile)
endif
$(_f90_depfile): $(top_srcdir)/fdep/fortran_dependencies.pl $(foreach p,$(bin_PROGRAMS),$(_$p_use_mods) $(_$p_def_mods)) | $(foreach p,$(bin_PROGRAMS),$(_f90_moddir)/$p)
	$(call _f90_verbose,F90 DEPS $@)echo > $@; $(foreach p,$(bin_PROGRAMS),$(top_srcdir)/fdep/fortran_dependencies.pl $(_$p_use_mods) $(_$p_def_mods) >> $@; )

$(_f90_moddir):
	@mkdir $@

$(foreach p,$(bin_PROGRAMS),$(_f90_moddir)/$p): | $(_f90_moddir)
	@mkdir $@

CLEANFILES += $(foreach p,$(bin_PROGRAMS),$(_$p_def_mods) $(_$p_use_mods))
CLEANFILES += $(foreach p,$(bin_PROGRAMS),$(_f90_moddir)/$p/*)
CLEANFILES += $(_f90_depfile)
