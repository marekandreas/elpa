#!/usr/bin/perl -w

use strict;

my %defs = ();
my %uses = ();

my $use_re = qr/^\s*use\s+(\S+)\s*$/;
my $def_re = qr/^\s*module\s+(\S+)\s*$/;

sub add_use {
	my ($file, $module) = @_;
	if (defined($defs{$module}) && $defs{$module} eq $file) {
		# do not add self-dependencies
		return;
	}
	if (!defined($uses{$file})) {
		$uses{$file} = { $module => 1 };
	} else {
		$uses{$file}{$module} = 1;
	}
}

sub add_def {
	my ($file, $module) = @_;
	if (!defined($defs{$module})) {
		$defs{$module} = $file;
		if (defined($uses{$file}) && defined($uses{$file}{$module})) {
			delete $uses{$file}{$module};
		}
	} else {
		die "Module $module both defined in $file, $defs{$module}";
	}
}

my $p = shift;

foreach my $file (@ARGV) {
	my $re;
	my $add;
	my $object;
	if (defined($ENV{V}) && $ENV{V} ge "2") {
		print STDERR "fdep: Considering file $file\n";
	}
	if ($file =~ /^(.*)\.def_mods.$p(\..*)$/) {
		$re = $def_re;
		$add = \&add_def;
		$object = $1 . $2;
	} elsif ($file =~ /^(.*)\.use_mods.$p(\..*)$/) {
		$re = $use_re;
		$add = \&add_use;
		$object = $1 . $2;
	} else {
		die "Unrecognized file extension for '$file'\nExpected (.*)\.def_mods.$p(\..*) or (.*)\.use_mods.$p(\..*)";
	}
	open(FILE,"<",$file) || die "\nCan't open $file: $!\n\n";
	while(<FILE>) {
		chomp;
		$_ = lc($_);
		if ($_ =~ $re) {
			&$add($object, $1);
		} else {
			die "Cannot parse module statement '$_', was expecting $re";
		}
	}
	close(FILE)
}

foreach my $object (sort keys %uses) {
	for my $m (keys %{$uses{$object}}) {
		if (defined $defs{$m}) {
			print "$object: ", $defs{$m}, "\n";
		} elsif (defined($ENV{V}) && $ENV{V} ge "1") {
			print STDERR "fdep: Warning: Cannot find definition of module $m in files for program $p, might be external\n";
		}
	}
}
